import { useState } from "react";
import {
  X,
  Shield,
  Lock,
  Ban,
  Code2,
  UserCog,
  User,
  ChevronRight,
  ArrowRight,
} from "lucide-react";
import { isElectron } from "../../utils/providerHelpers";

interface WelcomeModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// Broadcast on the window when onboarding picks the admin role. AppShell renders
// this modal indirectly (via the Playground) and resolves the launch view once on
// mount, so it can't observe `setAdmin` through props — it listens for this event
// instead and navigates admins to the Dashboard without waiting for a restart.
export const ADMIN_ROLE_CHOSEN_EVENT = "kiji:admin-role-chosen";

const promises = [
  {
    icon: Lock,
    title: "100% local",
    desc: "PII detection runs on your device.",
  },
  {
    icon: Ban,
    title: "No 3rd-party",
    desc: "Data only goes to your chosen AI.",
  },
  {
    icon: Code2,
    title: "Open source",
    desc: "Inspect and verify every claim.",
  },
];

export default function WelcomeModal({ isOpen, onClose }: WelcomeModalProps) {
  // "role" = ask whether the user is an admin; "welcome" = show the explainer.
  const [step, setStep] = useState<"role" | "welcome">("role");

  if (!isOpen) return null;

  // Persist the one-time dismissal so the welcome only shows on first launch.
  const dismiss = async () => {
    if (isElectron && window.electronAPI) {
      try {
        await window.electronAPI.setWelcomeDismissed(true);
      } catch (error) {
        console.error("Failed to save welcome dismissed preference:", error);
      }
    }
    onClose();
  };

  // Admins are setting Kiji up for others and don't need the explainer.
  const handleAdmin = async () => {
    if (isElectron && window.electronAPI) {
      try {
        await window.electronAPI.setAdmin(true);
        // Let AppShell re-resolve the launch view now that the role is known.
        window.dispatchEvent(new Event(ADMIN_ROLE_CHOSEN_EVENT));
      } catch (error) {
        console.error("Failed to save admin preference:", error);
      }
    }
    await dismiss();
  };

  return (
    <div className="fixed inset-0 bg-brand-950/40 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-lg w-full flex flex-col animate-rise-in">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-stone-200">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
              <Shield className="w-5 h-5" />
            </div>
            <h2 className="text-base font-semibold text-brand-900 tracking-tight">
              Welcome to Kiji Privacy Proxy
            </h2>
          </div>
          <button
            onClick={dismiss}
            className="p-1 text-stone-400 hover:text-stone-600 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6">
          {step === "role" ? (
            <div className="space-y-4">
              <p className="text-[13px] text-stone-500">How are you using Kiji?</p>
              <button
                onClick={handleAdmin}
                className="group w-full flex items-center gap-3 text-left p-4 rounded-xl ring-1 ring-stone-200 hover:bg-stone-50 hover:ring-brand-200 transition-colors"
              >
                <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
                  <UserCog className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <div className="text-sm font-semibold text-stone-900">
                    I'm an admin
                  </div>
                  <div className="text-[13px] text-stone-500">
                    Setting up Kiji for others
                  </div>
                </div>
                <ChevronRight className="w-4 h-4 text-stone-300 group-hover:text-brand-500 transition-colors" />
              </button>
              <button
                onClick={() => setStep("welcome")}
                className="group w-full flex items-center gap-3 text-left p-4 rounded-xl ring-1 ring-stone-200 hover:bg-stone-50 hover:ring-brand-200 transition-colors"
              >
                <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
                  <User className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <div className="text-sm font-semibold text-stone-900">
                    I'm a user
                  </div>
                  <div className="text-[13px] text-stone-500">
                    Using Kiji myself
                  </div>
                </div>
                <ChevronRight className="w-4 h-4 text-stone-300 group-hover:text-brand-500 transition-colors" />
              </button>
            </div>
          ) : (
            <div className="space-y-5">
              <p className="text-sm text-stone-600 leading-relaxed">
                Kiji masks personal data in your prompts before they reach any
                AI provider — all on your device.
              </p>
              <div className="grid grid-cols-3 gap-3">
                {promises.map(({ icon: Icon, title, desc }) => (
                  <div
                    key={title}
                    className="rounded-xl ring-1 ring-stone-200 p-3.5"
                  >
                    <div className="w-8 h-8 rounded-lg bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 mb-2.5">
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="text-[13px] font-semibold text-stone-900">
                      {title}
                    </div>
                    <div className="text-[11px] text-stone-500 leading-snug mt-0.5">
                      {desc}
                    </div>
                  </div>
                ))}
              </div>
              <button
                onClick={dismiss}
                className="btn-brand inline-flex w-full items-center justify-center gap-2 px-5 py-2.5 text-white rounded-xl text-sm font-medium tracking-tight"
              >
                Get Started
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
