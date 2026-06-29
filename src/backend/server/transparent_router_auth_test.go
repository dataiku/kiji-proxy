package server

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
	"testing"
)

// TestTransparentRouterAdminPathsAreProtected guards against a route being added
// to startTransparentProxy's path switch without also being registered as a
// Basic-Auth-protected path. The transparent listener is default-ALLOW (only
// paths in isTransparentAdminProtectedPath are challenged), so a new admin/data
// case that is not also listed there would silently ship unauthenticated.
//
// The test reads the explicit case values out of the switch via the AST (so a
// newly added route is picked up automatically) and asserts the invariant: any
// switch case that is not a public path (health/version probes, /api/pii/*, or
// proxy traffic per isBasicAuthPublicPath) must be reported as protected by
// isTransparentAdminProtectedPath.
func TestTransparentRouterAdminPathsAreProtected(t *testing.T) {
	cases := transparentRouterSwitchPaths(t)
	if len(cases) == 0 {
		t.Fatal("found no case paths in startTransparentProxy's router switch; did the function move?")
	}

	for _, p := range cases {
		if isBasicAuthPublicPath(p) {
			continue // intentionally open (health/version/pii)
		}
		if !isTransparentAdminProtectedPath(p) {
			t.Errorf("transparent router serves %q but it is not in isTransparentAdminProtectedPath; "+
				"it would be reachable without Basic Auth. Add it to isTransparentAdminProtectedPath.", p)
		}
	}
}

// transparentRouterSwitchPaths parses server.go and returns the string literals
// used as case values in the `switch r.URL.Path` inside startTransparentProxy.
func transparentRouterSwitchPaths(t *testing.T) []string {
	t.Helper()

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "server.go", nil, 0)
	if err != nil {
		t.Fatalf("parse server.go: %v", err)
	}

	var fn *ast.FuncDecl
	for _, decl := range file.Decls {
		if d, ok := decl.(*ast.FuncDecl); ok && d.Name.Name == "startTransparentProxy" {
			fn = d
			break
		}
	}
	if fn == nil {
		t.Fatal("could not find startTransparentProxy in server.go")
	}

	var paths []string
	ast.Inspect(fn, func(n ast.Node) bool {
		sw, ok := n.(*ast.SwitchStmt)
		if !ok || !isReqURLPathSelector(sw.Tag) {
			return true
		}
		for _, stmt := range sw.Body.List {
			clause, ok := stmt.(*ast.CaseClause)
			if !ok {
				continue
			}
			for _, expr := range clause.List {
				lit, ok := expr.(*ast.BasicLit)
				if !ok || lit.Kind != token.STRING {
					continue
				}
				if s, err := strconv.Unquote(lit.Value); err == nil {
					paths = append(paths, s)
				}
			}
		}
		return true
	})
	return paths
}

// isReqURLPathSelector reports whether expr is the `r.URL.Path` selector used as
// the switch tag in the transparent router.
func isReqURLPathSelector(expr ast.Expr) bool {
	path, ok := expr.(*ast.SelectorExpr)
	if !ok || path.Sel.Name != "Path" {
		return false
	}
	url, ok := path.X.(*ast.SelectorExpr)
	if !ok || url.Sel.Name != "URL" {
		return false
	}
	r, ok := url.X.(*ast.Ident)
	return ok && r.Name == "r"
}
