import type { TourOptions } from "shepherd.js";

export const tourOptions: TourOptions = {
  defaultStepOptions: {
    cancelIcon: { enabled: true },
    scrollTo: { behavior: "smooth", block: "center" },
    modalOverlayOpeningPadding: 8,
    modalOverlayOpeningRadius: 12,
  },
  useModalOverlay: true,
  exitOnEsc: true,
  keyboardNavigation: true,
};
