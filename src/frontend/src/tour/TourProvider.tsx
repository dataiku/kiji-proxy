import { ReactNode } from "react";
import "shepherd.js/dist/css/shepherd.css";
import "./shepherd-theme.css";

interface TourProviderProps {
  children: ReactNode;
}

export default function TourProvider({ children }: TourProviderProps) {
  return <>{children}</>;
}
