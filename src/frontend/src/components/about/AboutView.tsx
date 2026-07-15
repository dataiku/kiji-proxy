import { useEffect, useState } from "react";
import { useTranslation, Trans } from "react-i18next";
import {
  Code2,
  BookOpen,
  Puzzle,
  Bug,
  Lightbulb,
  Mail,
  Scale,
  ArrowUpRight,
} from "lucide-react";
import logoImage from "../../../assets/kiji_proxy.svg";
import { GO_SERVER_ADDRESS } from "../../utils/providerHelpers";

interface AboutLink {
  key: string;
  href: string;
  icon: typeof Code2;
  /** Plain mailto/anchors don't get target/rel. */
  external?: boolean;
}

const LINKS: AboutLink[] = [
  {
    key: "github",
    href: "https://github.com/dataiku/kiji-proxy/",
    icon: Code2,
    external: true,
  },
  {
    key: "docs",
    href: "https://github.com/dataiku/kiji-proxy/blob/main/docs/README.md",
    icon: BookOpen,
    external: true,
  },
  {
    key: "extension",
    href: "https://chromewebstore.google.com/detail/kiji-privacy-proxy-extens/knnjemahdeioghdgcpeikepmlajfihin",
    icon: Puzzle,
    external: true,
  },
  {
    key: "bug",
    href: "https://github.com/dataiku/kiji-proxy/issues/new?template=10_bug_report.yml",
    icon: Bug,
    external: true,
  },
  {
    key: "feature",
    href: "https://github.com/dataiku/kiji-proxy/discussions/new/choose",
    icon: Lightbulb,
    external: true,
  },
  {
    key: "email",
    href: "mailto:opensource@dataiku.com?subject=[Kiji Privacy Proxy User]",
    icon: Mail,
  },
  {
    key: "license",
    href: "https://github.com/dataiku/kiji-proxy/blob/main/LICENSE",
    icon: Scale,
    external: true,
  },
];

export default function AboutView() {
  const { t } = useTranslation("about");
  const [version, setVersion] = useState<string>(t("versionLoading"));

  useEffect(() => {
    const loadVersion = async () => {
      try {
        const response = await fetch(`${GO_SERVER_ADDRESS}/api/version`);
        if (response.ok) {
          const data = await response.json();
          setVersion(data.version || t("versionUnknown"));
        } else {
          setVersion(t("versionUnknown"));
        }
      } catch (error) {
        console.error("Failed to fetch version:", error);
        setVersion(t("versionUnknown"));
      }
    };
    loadVersion();
  }, [t]);

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-[23px] font-semibold tracking-tight text-stone-900">
          {t("title")}
        </h1>
        <p className="text-stone-500 text-[13px] mt-0.5">{t("subtitle")}</p>
      </div>

      <div className="space-y-4 animate-rise-in">
        {/* Hero: logo, name, version */}
        <section className="card p-6 md:p-7">
          <div className="flex flex-col items-center text-center sm:flex-row sm:items-center sm:text-left sm:gap-6">
            <img
              src={logoImage}
              alt={t("logoAlt")}
              className="w-20 h-20 shrink-0"
            />
            <div className="mt-4 sm:mt-0">
              <h2 className="text-2xl font-bold text-brand-900 tracking-tight">
                {t("productName")}
              </h2>
              <p className="text-stone-600 mt-1">{t("tagline")}</p>
              <div className="mt-3 inline-flex items-center gap-2 rounded-full bg-stone-100 px-3 py-1 ring-1 ring-stone-200">
                <span className="text-xs font-medium text-stone-500">
                  {t("versionLabel")}
                </span>
                <span className="text-xs font-mono font-semibold text-stone-800">
                  {version}
                </span>
              </div>
            </div>
          </div>

          <p className="mt-6 text-sm leading-relaxed text-stone-600">
            {t("description")}
          </p>
        </section>

        {/* Resources / links */}
        <section className="card p-6 md:p-7">
          <h3 className="text-base font-semibold text-brand-900 tracking-tight mb-4">
            {t("resources")}
          </h3>
          <div className="grid gap-2 sm:grid-cols-2">
            {LINKS.map(({ key, href, icon: Icon, external }) => (
              <a
                key={key}
                href={href}
                {...(external
                  ? { target: "_blank", rel: "noopener noreferrer" }
                  : {})}
                className="group flex items-center justify-between gap-3 rounded-xl ring-1 ring-stone-200 p-3.5 text-left hover:ring-brand-200 hover:bg-brand-50/40 transition-all"
              >
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-lg bg-stone-100 group-hover:bg-brand-50 flex items-center justify-center text-stone-600 group-hover:text-brand-600 transition-colors shrink-0">
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="font-medium text-stone-700 group-hover:text-brand-800 transition-colors">
                    {t(`links.${key}`)}
                  </span>
                </div>
                <ArrowUpRight className="w-4 h-4 text-stone-400 group-hover:text-brand-500 transition-colors shrink-0" />
              </a>
            ))}
          </div>
        </section>

        {/* No support obligations + copyright */}
        <section className="card p-6 md:p-7">
          <h3 className="text-base font-semibold text-brand-900 tracking-tight">
            {t("noSupport.title")}
          </h3>
          <p className="mt-2 text-sm leading-relaxed text-stone-600">
            {t("noSupport.body")}
          </p>

          <div className="mt-5 pt-5 border-t border-stone-200 text-xs text-stone-500 leading-relaxed">
            <Trans
              t={t}
              i18nKey="copyright"
              values={{ year: new Date().getFullYear() }}
              components={{
                labLink: (
                  <a
                    href="https://www.dataiku.com/company/dataiku-for-the-future/open-source/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-brand-600 hover:text-brand-700 hover:underline"
                  />
                ),
              }}
            />
          </div>
        </section>
      </div>
    </div>
  );
}
