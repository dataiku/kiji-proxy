import type { StepOptions } from "shepherd.js";
import { isElectron } from "../utils/providerHelpers";

export function getTourSteps(): StepOptions[] {
  const steps: StepOptions[] = [
    {
      id: "welcome",
      attachTo: { element: "#tour-header", on: "bottom" },
      title: "Welcome to Yaak Privacy Proxy",
      text: "This is your privacy-first proxy for LLM requests. It detects and masks personal information before sending your data to AI providers.",
      buttons: [
        {
          text: "Skip",
          secondary: true,
          action() {
            this.cancel();
          },
        },
        {
          text: "Next",
          action() {
            this.next();
          },
        },
      ],
      cancelIcon: { enabled: true },
    },
  ];

  if (isElectron) {
    steps.push({
      id: "provider-selector",
      attachTo: { element: "#tour-provider-selector", on: "bottom" },
      title: "Choose Your AI Provider",
      text: "Select which AI provider to send your requests to. Configure API keys for each provider in Settings.",
      buttons: [
        {
          text: "Back",
          secondary: true,
          action() {
            this.back();
          },
        },
        {
          text: "Next",
          action() {
            this.next();
          },
        },
      ],
      cancelIcon: { enabled: true },
    });
  }

  steps.push(
    {
      id: "input-section",
      attachTo: { element: "#tour-input-section", on: "bottom" },
      title: "Enter Your Message",
      text: "Type or paste your message here. It can contain sensitive information like names, emails, or phone numbers \u2014 Yaak will detect and mask them automatically.",
      buttons: [
        {
          text: "Back",
          secondary: true,
          action() {
            this.back();
          },
        },
        {
          text: "Next",
          action() {
            this.next();
          },
        },
      ],
      cancelIcon: { enabled: true },
    },
    {
      id: "process-button",
      attachTo: { element: "#tour-process-button", on: "right" },
      title: "Process Your Data",
      text: "Click this button to send your message through the privacy proxy. Yaak will detect PII, replace it with realistic fake data, send the masked version to the AI, and restore your original data in the response.",
      buttons: [
        {
          text: "Back",
          secondary: true,
          action() {
            this.back();
          },
        },
        {
          text: "Next",
          action() {
            this.next();
          },
        },
      ],
      cancelIcon: { enabled: true },
    }
  );

  if (isElectron) {
    steps.push({
      id: "menu-button",
      attachTo: { element: "#tour-menu-button", on: "right" },
      title: "Access Settings & More",
      text: "Open this menu to configure API keys in Settings, view request logs in Logging, or learn more About the app. You can also restart this tour from here.",
      buttons: [
        {
          text: "Back",
          secondary: true,
          action() {
            this.back();
          },
        },
        {
          text: "Next",
          action() {
            this.next();
          },
        },
      ],
      cancelIcon: { enabled: true },
    });
  }

  steps.push({
    id: "status-bar",
    attachTo: { element: "#tour-status-bar", on: "top" },
    title: "Server Status",
    text: "This bar shows whether the backend server is running. A green dot means the PII detection engine is ready. The model signature verifies the integrity of the ML model.",
    buttons: [
      {
        text: "Back",
        secondary: true,
        action() {
          this.back();
        },
      },
      {
        text: "Finish",
        action() {
          this.complete();
        },
      },
    ],
    cancelIcon: { enabled: true },
  });

  return steps;
}
