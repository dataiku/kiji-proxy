import { useCallback, useEffect, useRef, useState } from "react";
import { useShepherd } from "react-shepherd";
import { getTourSteps } from "./tourSteps";
import { tourOptions } from "./tourOptions";

const isElectron =
  typeof window !== "undefined" && window.electronAPI !== undefined;

const TOUR_COMPLETED_KEY = "yaak-tour-completed";

async function loadTourCompleted(): Promise<boolean> {
  if (isElectron && window.electronAPI) {
    return window.electronAPI.getTourCompleted();
  }
  try {
    return localStorage.getItem(TOUR_COMPLETED_KEY) === "true";
  } catch {
    return false;
  }
}

function saveTourCompleted(completed: boolean): void {
  if (isElectron && window.electronAPI) {
    window.electronAPI.setTourCompleted(completed);
    return;
  }
  try {
    localStorage.setItem(TOUR_COMPLETED_KEY, String(completed));
  } catch {
    // localStorage may not be available
  }
}

interface TourInstance {
  start: () => Promise<void>;
  cancel: () => Promise<void>;
  isActive: () => boolean;
  on: (event: string, handler: () => void) => void;
  addSteps: (steps: ReturnType<typeof getTourSteps>) => void;
}

export function useTour(
  welcomeModalJustClosed: boolean,
  termsAccepted: boolean
) {
  const Shepherd = useShepherd();
  const tourRef = useRef<TourInstance | null>(null);
  const hasAutoStarted = useRef(false);
  const [tourCompleted, setTourCompleted] = useState<boolean | null>(null);

  // Load tour-completed state on mount
  useEffect(() => {
    loadTourCompleted().then(setTourCompleted);
  }, []);

  // Mark tour as completed and persist
  const markTourCompleted = useCallback(() => {
    setTourCompleted(true);
    saveTourCompleted(true);
  }, []);

  // Create or retrieve the tour instance
  const getTour = useCallback((): TourInstance => {
    if (!tourRef.current) {
      const tour = new Shepherd.Tour(tourOptions) as unknown as TourInstance;
      tour.addSteps(getTourSteps());

      tour.on("complete", () => markTourCompleted());
      tour.on("cancel", () => markTourCompleted());

      tourRef.current = tour;
    }
    return tourRef.current;
  }, [Shepherd, markTourCompleted]);

  // Auto-start after WelcomeModal closes and terms are accepted (first time only)
  useEffect(() => {
    if (
      welcomeModalJustClosed &&
      termsAccepted &&
      tourCompleted === false &&
      !hasAutoStarted.current
    ) {
      hasAutoStarted.current = true;
      const timer = setTimeout(() => {
        getTour().start();
      }, 500);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [welcomeModalJustClosed, termsAccepted, tourCompleted, getTour]);

  // Manual start (from menu) — always works regardless of completion state
  const startTour = useCallback(() => {
    const currentTour = tourRef.current;
    if (currentTour && currentTour.isActive()) {
      currentTour.cancel();
    }
    // Recreate steps in case environment changed
    tourRef.current = null;
    getTour().start();
  }, [getTour]);

  // Cancel if component unmounts
  useEffect(() => {
    return () => {
      if (tourRef.current && tourRef.current.isActive()) {
        tourRef.current.cancel();
      }
    };
  }, []);

  // Check if tour is active (for use in effects/event handlers only)
  const isTourActive = useCallback(() => {
    return tourRef.current?.isActive() ?? false;
  }, []);

  const cancelTour = useCallback(() => {
    if (tourRef.current && tourRef.current.isActive()) {
      tourRef.current.cancel();
    }
  }, []);

  return { startTour, isTourActive, cancelTour };
}
