import { ReactNode } from "react";
import { ShepherdJourneyProvider } from "react-shepherd";
import "shepherd.js/dist/css/shepherd.css";
import "./shepherd-theme.css";

interface TourProviderProps {
  children: ReactNode;
}

export default function TourProvider({ children }: TourProviderProps) {
  return <ShepherdJourneyProvider>{children}</ShepherdJourneyProvider>;
}
