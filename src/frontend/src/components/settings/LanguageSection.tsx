import { useTranslation } from "react-i18next";
import { Languages } from "lucide-react";
import { SUPPORTED_LANGUAGES } from "../../i18n";

/**
 * Language preference selector.
 *
 * Writes through i18next's language detector, which is configured with
 * `caches: ["localStorage"]` — so the choice persists across restarts without
 * any extra storage code here. Syncing the selection to the Electron main
 * process (for the native menu) is handled separately (#578).
 */
export default function LanguageSection() {
  const { t, i18n } = useTranslation("common");

  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    void i18n.changeLanguage(event.target.value);
  };

  return (
    <section className="card p-6 md:p-7">
      {/* Section header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
          <Languages className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-brand-900 tracking-tight">
            {t("language.label")}
          </h2>
          <p className="text-[13px] text-stone-500">{t("language.subtitle")}</p>
        </div>
      </div>

      <label className="sr-only" htmlFor="kiji-language-select">
        {t("language.label")}
      </label>
      <select
        id="kiji-language-select"
        value={i18n.resolvedLanguage}
        onChange={handleChange}
        className="w-full sm:w-64 px-3 py-2 rounded-lg border border-stone-200 bg-white text-sm transition-shadow focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
      >
        {SUPPORTED_LANGUAGES.map((lng) => (
          <option key={lng} value={lng}>
            {t(`language.${lng}`)}
          </option>
        ))}
      </select>
    </section>
  );
}
