import { Component, ErrorInfo, ReactNode } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { reportError } from "../utils/misclassificationReporter";
import { withTranslation, WithTranslation } from "react-i18next";

interface Props extends WithTranslation {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI.
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error details to console
    console.error("Error caught by ErrorBoundary:", error);
    console.error("Error info:", errorInfo);

    // Update state with error details
    this.setState({
      error,
      errorInfo,
    });

    // Forward to Sentry. This is a no-op when telemetry is disabled (opt-in),
    // so no data leaves the machine unless the user turned reporting on.
    reportError(error, { componentStack: errorInfo.componentStack });
  }

  handleReset = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    const { t } = this.props;
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-red-50 to-red-100 flex items-center justify-center p-4">
          <div className="max-w-2xl w-full bg-white rounded-xl shadow-2xl overflow-hidden">
            {/* Header */}
            <div className="bg-red-600 text-white p-6">
              <div className="flex items-center gap-3">
                <AlertCircle className="w-8 h-8" />
                <div>
                  <h1 className="text-2xl font-bold">{t("error.title")}</h1>
                  <p className="text-red-100 mt-1">{t("error.subtitle")}</p>
                </div>
              </div>
            </div>

            {/* Error Details */}
            <div className="p-6 space-y-4">
              {this.state.error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <h2 className="text-sm font-semibold text-red-900 mb-2">
                    {t("error.messageLabel")}
                  </h2>
                  <p className="text-sm text-red-800 font-mono">
                    {this.state.error.message}
                  </p>
                </div>
              )}

              {this.state.errorInfo && (
                <details className="bg-stone-50 border border-stone-200 rounded-lg p-4">
                  <summary className="text-sm font-semibold text-stone-900 cursor-pointer hover:text-stone-700">
                    {t("error.stackTrace")}
                  </summary>
                  <pre className="mt-3 text-xs text-stone-700 font-mono overflow-x-auto whitespace-pre-wrap">
                    {this.state.errorInfo.componentStack}
                  </pre>
                </details>
              )}

              {/* Actions */}
              <div className="flex gap-3 pt-4">
                <button
                  onClick={this.handleReset}
                  className="flex items-center gap-2 px-6 py-3 bg-brand-600 text-white rounded-lg hover:bg-brand-700 transition-colors font-medium"
                >
                  <RefreshCw className="w-5 h-5" />
                  {t("error.tryAgain")}
                </button>
                <button
                  onClick={this.handleReload}
                  className="px-6 py-3 border-2 border-stone-300 text-stone-700 rounded-lg hover:bg-stone-50 transition-colors font-medium"
                >
                  {t("error.reload")}
                </button>
              </div>

              {/* Help Text */}
              <div className="mt-6 p-4 bg-brand-50 border border-brand-200 rounded-lg">
                <h3 className="text-sm font-semibold text-brand-900 mb-2">
                  {t("error.whatCanYouDo")}
                </h3>
                <ul className="text-sm text-brand-800 space-y-1 list-disc list-inside">
                  <li>{t("error.help.tryAgain")}</li>
                  <li>{t("error.help.reload")}</li>
                  <li>{t("error.help.console")}</li>
                  <li>{t("error.help.report")}</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default withTranslation("common")(ErrorBoundary);
