import { useState, useEffect, CSSProperties, ReactNode } from "react";
import { X, Shield, ChevronLeft, ChevronRight } from "lucide-react";

/* ═══════════════════════════════════════════════════════════════════════════
   THEME CONSTANTS
   ═══════════════════════════════════════════════════════════════════════════ */

const colors = {
  blue: "#6C9CFF",
  green: "#34D399",
  purple: "#A78BFA",
  orange: "#F59E42",
  red: "#F87171",
  text: "#E8ECF4",
  textHeading: "#F0F3FA",
  textMuted: "rgba(180,190,210,0.6)",
  textMutedLight: "rgba(180,190,210,0.7)",
  bgDark: "#0B0F1A",
};

const fonts = {
  heading: "'Outfit','DM Sans',sans-serif",
  body: "'Inter',sans-serif",
  mono: "'JetBrains Mono','SF Mono',monospace",
};

const gradients = {
  background: "linear-gradient(160deg, #070A12 0%, #0D1220 40%, #0F1628 100%)",
  blueToGreen: "linear-gradient(135deg, #34D399, #6C9CFF)",
  blueToPurple: "linear-gradient(135deg, #6C9CFF, #A78BFA)",
};

const transitions = {
  default: "all 0.6s cubic-bezier(0.4,0,0.2,1)",
  fast: "all 0.3s ease",
};

/* ═══════════════════════════════════════════════════════════════════════════
   SHARED STYLES
   ═══════════════════════════════════════════════════════════════════════════ */

const gridPatternStyle: CSSProperties = {
  position: "absolute",
  inset: 0,
  backgroundImage: `
    linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px)
  `,
  backgroundSize: "40px 40px",
  pointerEvents: "none",
};

const keyframeStyles = `
  @keyframes floatDot {
    0% { transform: translate(0, 0); }
    100% { transform: translate(8px, -12px); }
  }
  @keyframes scrollTicker {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
  }
`;

/* ═══════════════════════════════════════════════════════════════════════════
   REUSABLE COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

/* ───── Pane Container (dark background with grid) ───── */
function PaneContainer({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        background: gradients.background,
        fontFamily: fonts.heading,
        padding: 20,
        borderRadius: 12,
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div style={gridPatternStyle} />
      <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
      <style>{keyframeStyles}</style>
    </div>
  );
}

/* ───── Ambient Glow Effect ───── */
function AmbientGlow({
  color,
  size,
  position,
}: {
  color: string;
  size: number;
  position: { top?: string; bottom?: string; left?: string; right?: string };
}) {
  return (
    <div
      style={{
        position: "absolute",
        ...position,
        width: size,
        height: size,
        borderRadius: "50%",
        background: `radial-gradient(circle, ${color} 0%, transparent 70%)`,
        pointerEvents: "none",
      }}
    />
  );
}

/* ───── Section Header (colored dot + label) ───── */
function SectionHeader({
  color,
  label,
  visible,
  delay = 0,
}: {
  color: string;
  label: string;
  visible: boolean;
  delay?: number;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        marginBottom: 8,
        opacity: visible ? 1 : 0,
        transition: `opacity 0.5s ${delay}s`,
      }}
    >
      <div
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: color,
          boxShadow: `0 0 8px ${color}40`,
        }}
      />
      <span
        style={{
          fontSize: 10,
          fontWeight: 600,
          color,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          fontFamily: fonts.mono,
        }}
      >
        {label}
      </span>
    </div>
  );
}

/* ───── Trust Badge (shield icon + text) ───── */
function TrustBadge({
  text,
  visible,
  delay = 0,
  size = 12,
}: {
  text: string;
  visible: boolean;
  delay?: number;
  size?: number;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 6,
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(8px)",
        transition: `${transitions.default} ${delay}s`,
      }}
    >
      <svg width={size} height={size} viewBox="0 0 20 20" fill="none">
        <path
          d="M10 1L3 4.5V9.5C3 14.2 6 17.5 10 19C14 17.5 17 14.2 17 9.5V4.5L10 1Z"
          stroke={colors.green}
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
        <path
          d="M7 10l2 2 4-4"
          stroke={colors.green}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span
        style={{
          fontSize: size - 2,
          fontWeight: 500,
          color: `${colors.green}B3`,
          letterSpacing: "0.04em",
        }}
      >
        {text}
      </span>
    </div>
  );
}

/* ───── Pane Title ───── */
function PaneTitle({
  children,
  subtitle,
  visible,
}: {
  children: ReactNode;
  subtitle?: string;
  visible: boolean;
}) {
  return (
    <div
      style={{
        textAlign: "center",
        marginBottom: 20,
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(-16px)",
        transition: "all 0.7s cubic-bezier(0.4,0,0.2,1)",
      }}
    >
      <h2
        style={{
          fontSize: 20,
          fontWeight: 700,
          color: colors.textHeading,
          margin: "0 0 6px",
          letterSpacing: "-0.025em",
          lineHeight: 1.2,
        }}
      >
        {children}
      </h2>
      {subtitle && (
        <p
          style={{
            fontSize: 12,
            color: colors.textMuted,
            margin: 0,
            lineHeight: 1.5,
          }}
        >
          {subtitle}
        </p>
      )}
    </div>
  );
}

/* ───── Gradient Text ───── */
function GradientText({
  children,
  gradient = gradients.blueToGreen,
}: {
  children: ReactNode;
  gradient?: string;
}) {
  return (
    <span
      style={{
        background: gradient,
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
      }}
    >
      {children}
    </span>
  );
}

/* ───── Hover Card (with glow effect) ───── */
function HoverCard({
  children,
  color,
  visible,
  delay,
  padding = "12px 14px",
}: {
  children: ReactNode;
  color: string;
  visible: boolean;
  delay: number;
  padding?: string;
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 12,
        padding,
        borderRadius: 12,
        background: hovered ? `${color}08` : "rgba(255,255,255,0.02)",
        border: `1px solid ${
          hovered ? color + "30" : "rgba(255,255,255,0.05)"
        }`,
        opacity: visible ? 1 : 0,
        transform: visible ? "translateX(0)" : "translateX(-20px)",
        transition: `${transitions.default} ${delay}s, background 0.3s, border 0.3s`,
        cursor: "default",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Glow effect */}
      <div
        style={{
          position: "absolute",
          top: -20,
          left: -20,
          width: 60,
          height: 60,
          borderRadius: "50%",
          background: color,
          opacity: hovered ? 0.06 : 0,
          filter: "blur(25px)",
          transition: "opacity 0.4s",
          pointerEvents: "none",
        }}
      />
      {children}
    </div>
  );
}

/* ───── Animated Floating Particles ───── */
const particleDots = Array.from({ length: 18 }, (_, i) => ({
  id: i,
  x: (i * 37 + 13) % 100,
  y: (i * 53 + 7) % 100,
  size: 1.5 + ((i * 17) % 20) / 10,
  dur: 12 + ((i * 29) % 20),
  delay: -((i * 11) % 20),
  opacity: 0.06 + ((i * 13) % 8) / 100,
}));

function Particles() {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        pointerEvents: "none",
      }}
    >
      {particleDots.map((d) => (
        <div
          key={d.id}
          style={{
            position: "absolute",
            left: `${d.x}%`,
            top: `${d.y}%`,
            width: d.size,
            height: d.size,
            borderRadius: "50%",
            background: colors.blue,
            opacity: d.opacity,
            animation: `floatDot ${d.dur}s ease-in-out ${d.delay}s infinite alternate`,
          }}
        />
      ))}
    </div>
  );
}

/* ───── Icon Box ───── */
function IconBox({
  children,
  color,
  size = 38,
}: {
  children: ReactNode;
  color: string;
  size?: number;
}) {
  return (
    <div
      style={{
        flexShrink: 0,
        width: size,
        height: size,
        borderRadius: 10,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: `${color}10`,
        border: `1px solid ${color}25`,
        color,
      }}
    >
      {children}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   WORKFLOW STEP DATA & COMPONENTS (Pane 2)
   ═══════════════════════════════════════════════════════════════════════════ */

const workflowSteps = [
  {
    number: "01",
    title: "Send Request",
    description: "Your app sends a request to the Yaak proxy",
    icon: (
      <svg viewBox="0 0 40 40" fill="none" style={{ width: 40, height: 40 }}>
        <rect
          x="4"
          y="8"
          width="24"
          height="18"
          rx="3"
          stroke="currentColor"
          strokeWidth="2"
        />
        <path
          d="M12 20h8M16 16v8"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <path
          d="M30 16l6 4-6 4"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ),
    color: colors.blue,
  },
  {
    number: "02",
    title: "Detect PII",
    description:
      "Local ML model identifies personal data — no third parties involved",
    icon: (
      <svg viewBox="0 0 40 40" fill="none" style={{ width: 40, height: 40 }}>
        <circle cx="20" cy="16" r="8" stroke="currentColor" strokeWidth="2" />
        <path
          d="M14 32c0-3.3 2.7-6 6-6s6 2.7 6 6"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <path
          d="M28 12l4-4M32 12l-4-4"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
    ),
    color: colors.orange,
  },
  {
    number: "03",
    title: "Mask & Replace",
    description:
      "PII is swapped with realistic dummy data before the AI API call",
    icon: (
      <svg viewBox="0 0 40 40" fill="none" style={{ width: 40, height: 40 }}>
        <rect
          x="6"
          y="10"
          width="28"
          height="20"
          rx="4"
          stroke="currentColor"
          strokeWidth="2"
        />
        <path
          d="M12 18h6M12 24h10"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <circle cx="28" cy="18" r="2" fill="currentColor" />
        <circle cx="28" cy="24" r="2" fill="currentColor" />
      </svg>
    ),
    color: colors.purple,
  },
  {
    number: "04",
    title: "Restore Data",
    description:
      "The AI response arrives and original information is restored automatically",
    icon: (
      <svg viewBox="0 0 40 40" fill="none" style={{ width: 40, height: 40 }}>
        <path
          d="M20 6v8l4-3M20 14l-4-3"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M10 20a10 10 0 1 0 20 0"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <path
          d="M16 24l4 4 6-8"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ),
    color: colors.green,
  },
  {
    number: "05",
    title: "Deliver Response",
    description:
      "Your app receives the full response with original data intact",
    icon: (
      <svg viewBox="0 0 40 40" fill="none" style={{ width: 40, height: 40 }}>
        <rect
          x="8"
          y="6"
          width="24"
          height="28"
          rx="4"
          stroke="currentColor"
          strokeWidth="2"
        />
        <path
          d="M14 14h12M14 20h8M14 26h10"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <circle cx="30" cy="30" r="6" fill="currentColor" opacity="0.2" />
        <path
          d="M28 30l2 2 3-4"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ),
    color: colors.blue,
  },
];

function ArrowConnector({
  color,
  nextColor,
  animated,
  index,
}: {
  color: string;
  nextColor?: string;
  animated: boolean;
  index: number;
}) {
  const midColor = nextColor || color;
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: 32,
        opacity: animated ? 1 : 0,
        transform: animated ? "scaleY(1)" : "scaleY(0)",
        transition: `all 0.5s cubic-bezier(0.4,0,0.2,1) ${index * 0.18 + 0.3}s`,
        transformOrigin: "top center",
      }}
    >
      <svg width="40" height="32" viewBox="0 0 40 32" fill="none">
        <defs>
          <linearGradient
            id={`grad-${index}`}
            x1="20"
            y1="0"
            x2="20"
            y2="32"
            gradientUnits="userSpaceOnUse"
          >
            <stop stopColor={color} stopOpacity="0.6" />
            <stop offset="1" stopColor={midColor} stopOpacity="0.6" />
          </linearGradient>
        </defs>
        <line
          x1="20"
          y1="2"
          x2="20"
          y2="22"
          stroke={`url(#grad-${index})`}
          strokeWidth="2"
          strokeDasharray="4 4"
        />
        <path
          d="M14 20l6 8 6-8"
          fill="none"
          stroke={midColor}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity="0.7"
        />
      </svg>
    </div>
  );
}

function StepCard({
  step,
  index,
  animated,
}: {
  step: (typeof workflowSteps)[0];
  index: number;
  animated: boolean;
}) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 16,
        padding: "12px 16px",
        borderRadius: 12,
        background: hovered
          ? `linear-gradient(135deg, ${step.color}18 0%, ${step.color}0a 100%)`
          : "rgba(15,22,40,0.6)",
        border: `1px solid ${
          hovered ? step.color + "40" : "rgba(255,255,255,0.08)"
        }`,
        opacity: animated ? 1 : 0,
        transform: animated ? "translateY(0)" : "translateY(24px)",
        transition: `${transitions.default} ${
          index * 0.18
        }s, background 0.3s, border 0.3s`,
        cursor: "default",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: -30,
          left: -30,
          width: 80,
          height: 80,
          borderRadius: "50%",
          background: step.color,
          opacity: hovered ? 0.06 : 0,
          filter: "blur(30px)",
          transition: "opacity 0.4s",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          flexShrink: 0,
          width: 48,
          height: 48,
          borderRadius: 10,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: `linear-gradient(145deg, ${step.color}18, ${step.color}08)`,
          border: `1px solid ${step.color}30`,
          color: step.color,
          position: "relative",
        }}
      >
        <div style={{ width: 28, height: 28 }}>{step.icon}</div>
        <span
          style={{
            position: "absolute",
            top: -5,
            right: -5,
            fontSize: 9,
            fontWeight: 700,
            fontFamily: "monospace",
            background: step.color,
            color: colors.bgDark,
            borderRadius: 4,
            padding: "1px 4px",
            letterSpacing: "0.02em",
          }}
        >
          {step.number}
        </span>
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            color: colors.text,
            marginBottom: 2,
            letterSpacing: "-0.01em",
          }}
        >
          {step.title}
        </div>
        <div
          style={{
            fontSize: 12,
            color: colors.textMutedLight,
            lineHeight: 1.4,
            letterSpacing: "0.005em",
          }}
        >
          {step.description}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   PII DATA & COMPONENTS (Pane 1 & 3)
   ═══════════════════════════════════════════════════════════════════════════ */

const piiExamples = [
  { label: "john.doe@email.com", masked: "t.smith42@mail.net", type: "Email" },
  { label: "555-0123-4567", masked: "555-9876-1234", type: "Phone" },
  { label: "John Smith", masked: "Alex Turner", type: "Name" },
  { label: "4242 4242 4242 4242", masked: "5111 8833 2200 6677", type: "Card" },
  { label: "123-45-6789", masked: "987-65-4321", type: "SSN" },
];

const piiTypes = [
  { icon: "📧", label: "Emails", color: colors.blue },
  { icon: "📞", label: "Phone Numbers", color: colors.orange },
  { icon: "🆔", label: "SSNs", color: colors.red },
  { icon: "💳", label: "Credit Cards", color: colors.purple },
  { icon: "🌐", label: "IP Addresses", color: colors.green },
  { icon: "👤", label: "Names", color: colors.blue },
  { icon: "🏠", label: "Addresses", color: colors.orange },
  { icon: "📅", label: "Date of Birth", color: colors.red },
  { icon: "🏥", label: "Medical IDs", color: colors.purple },
  { icon: "🔑", label: "Passwords", color: colors.green },
  { icon: "🏦", label: "Bank Accounts", color: colors.blue },
  { icon: "🛂", label: "Passport Nos.", color: colors.orange },
  { icon: "🚗", label: "License Plates", color: colors.red },
  { icon: "🔐", label: "API Tokens", color: colors.purple },
  { icon: "💼", label: "Tax IDs", color: colors.green },
  { icon: "➕", label: "And More…", color: "rgba(255,255,255,0.3)" },
];

function PIITicker({ visible }: { visible: boolean }) {
  return (
    <div
      style={{
        overflow: "hidden",
        borderRadius: 10,
        border: "1px solid rgba(255,255,255,0.06)",
        background: "rgba(255,255,255,0.02)",
        opacity: visible ? 1 : 0,
        transition: "opacity 0.6s 0.8s",
      }}
    >
      <div
        style={{
          display: "flex",
          animation: "scrollTicker 20s linear infinite",
          width: "max-content",
        }}
      >
        {[...piiExamples, ...piiExamples].map((item, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "8px 14px",
              borderRight: "1px solid rgba(255,255,255,0.04)",
              whiteSpace: "nowrap",
            }}
          >
            <span
              style={{
                fontSize: 8,
                fontWeight: 600,
                fontFamily: fonts.mono,
                color: `${colors.red}90`,
                textDecoration: "line-through",
                letterSpacing: "0.02em",
              }}
            >
              {item.label}
            </span>
            <span style={{ color: "rgba(255,255,255,0.15)", fontSize: 10 }}>
              →
            </span>
            <span
              style={{
                fontSize: 8,
                fontWeight: 600,
                fontFamily: fonts.mono,
                color: `${colors.green}90`,
                letterSpacing: "0.02em",
              }}
            >
              {item.masked}
            </span>
            <span
              style={{
                fontSize: 7,
                fontWeight: 500,
                color: `${colors.purple}80`,
                fontFamily: fonts.body,
                background: `${colors.purple}10`,
                padding: "1px 4px",
                borderRadius: 3,
              }}
            >
              {item.type}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function PIIGrid({ visible }: { visible: boolean }) {
  return (
    <div
      style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6 }}
    >
      {piiTypes.map((pii, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 3,
            padding: "8px 3px 6px",
            borderRadius: 8,
            background: `${pii.color}06`,
            border: `1px solid ${pii.color}15`,
            opacity: visible ? 1 : 0,
            transform: visible ? "scale(1)" : "scale(0.85)",
            transition: `all 0.4s cubic-bezier(0.4,0,0.2,1) ${0.6 + i * 0.04}s`,
          }}
        >
          <span style={{ fontSize: 14 }}>{pii.icon}</span>
          <span
            style={{
              fontSize: 7.5,
              fontWeight: 500,
              color: `${pii.color}BB`,
              fontFamily: fonts.body,
              textAlign: "center",
              lineHeight: 1.2,
            }}
          >
            {pii.label}
          </span>
        </div>
      ))}
    </div>
  );
}

function StatBadge({
  icon,
  label,
  value,
  color,
  delay,
  visible,
}: {
  icon: string;
  label: string;
  value: string;
  color: string;
  delay: number;
  visible: boolean;
}) {
  return (
    <div
      style={{
        flex: "1 1 0",
        minWidth: 100,
        padding: "12px 10px",
        borderRadius: 12,
        background: `linear-gradient(145deg, ${color}0A, ${color}04)`,
        border: `1px solid ${color}20`,
        textAlign: "center",
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(16px)",
        transition: `${transitions.default} ${delay}s`,
      }}
    >
      <div style={{ fontSize: 18, marginBottom: 4 }}>{icon}</div>
      <div
        style={{
          fontFamily: fonts.heading,
          fontSize: 14,
          fontWeight: 700,
          color,
          letterSpacing: "-0.02em",
        }}
      >
        {value}
      </div>
      <div
        style={{
          fontFamily: fonts.body,
          fontSize: 10,
          color: `${color}99`,
          marginTop: 2,
          lineHeight: 1.3,
        }}
      >
        {label}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   PRIVACY PROMISES DATA (Pane 3)
   ═══════════════════════════════════════════════════════════════════════════ */

const privacyPromises = [
  {
    icon: (
      <svg viewBox="0 0 32 32" fill="none" style={{ width: 24, height: 24 }}>
        <rect
          x="4"
          y="6"
          width="24"
          height="20"
          rx="4"
          stroke="currentColor"
          strokeWidth="1.5"
        />
        <circle cx="16" cy="16" r="5" stroke="currentColor" strokeWidth="1.5" />
        <path
          d="M13 16l2 2 4-4"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="8" cy="10" r="1" fill="currentColor" opacity="0.5" />
        <circle cx="11" cy="10" r="1" fill="currentColor" opacity="0.5" />
      </svg>
    ),
    title: "100% Local Processing",
    desc: "All PII detection runs on your device. Nothing is sent externally.",
    color: colors.green,
  },
  {
    icon: (
      <svg viewBox="0 0 32 32" fill="none" style={{ width: 24, height: 24 }}>
        <circle
          cx="16"
          cy="16"
          r="11"
          stroke="currentColor"
          strokeWidth="1.5"
        />
        <path
          d="M16 5v3M16 24v3M5 16h3M24 16h3"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinecap="round"
          opacity="0.5"
        />
        <path
          d="M11 16l3 3 6-6"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ),
    title: "No Third-Party Sharing",
    desc: "Data only goes to your chosen AI provider — nowhere else.",
    color: colors.blue,
  },
  {
    icon: (
      <svg viewBox="0 0 32 32" fill="none" style={{ width: 24, height: 24 }}>
        <path
          d="M8 4h10l6 6v16a2 2 0 01-2 2H8a2 2 0 01-2-2V6a2 2 0 012-2z"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
        <path
          d="M18 4v6h6"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
        <path
          d="M10 18h12M10 22h8"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinecap="round"
          opacity="0.5"
        />
        <path
          d="M11 14l2 2 4-4"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ),
    title: "Open Source",
    desc: "Review the code, verify claims, and customize for your needs.",
    color: colors.purple,
  },
];

function PromiseCard({
  item,
  index,
  visible,
}: {
  item: (typeof privacyPromises)[0];
  index: number;
  visible: boolean;
}) {
  return (
    <HoverCard color={item.color} visible={visible} delay={1.3 + index * 0.12}>
      <IconBox color={item.color}>{item.icon}</IconBox>
      <div style={{ flex: 1 }}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: colors.text,
            fontFamily: fonts.heading,
            marginBottom: 2,
            letterSpacing: "-0.01em",
          }}
        >
          {item.title}
        </div>
        <div
          style={{
            fontSize: 11,
            color: colors.textMuted,
            fontFamily: fonts.body,
            lineHeight: 1.4,
          }}
        >
          {item.desc}
        </div>
      </div>
    </HoverCard>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   DIAGRAM COMPONENTS (Pane 1)
   ═══════════════════════════════════════════════════════════════════════════ */

function DangerDiagram({ visible }: { visible: boolean }) {
  const t = (delay: number) => ({ transition: `opacity 0.6s ${delay}s` });
  return (
    <svg
      viewBox="0 0 360 120"
      fill="none"
      style={{
        width: "100%",
        maxWidth: 360,
        height: "auto",
        display: "block",
        margin: "0 auto",
      }}
    >
      <rect
        x="4"
        y="30"
        width="90"
        height="60"
        rx="12"
        stroke={colors.red}
        strokeWidth="1.5"
        opacity={visible ? 0.7 : 0}
        style={t(0.1)}
      />
      <text
        x="49"
        y="56"
        textAnchor="middle"
        fill={colors.red}
        fontSize="10"
        fontWeight="600"
        fontFamily={fonts.heading}
        opacity={visible ? 1 : 0}
        style={t(0.2)}
      >
        Your App
      </text>
      <text
        x="49"
        y="72"
        textAnchor="middle"
        fill={`${colors.red}80`}
        fontSize="8"
        fontFamily={fonts.body}
        opacity={visible ? 1 : 0}
        style={t(0.25)}
      >
        + PII Data
      </text>
      <line
        x1="100"
        y1="60"
        x2="190"
        y2="60"
        stroke={colors.red}
        strokeWidth="1.2"
        strokeDasharray="5 4"
        opacity={visible ? 0.5 : 0}
        style={t(0.3)}
      />
      {visible && (
        <>
          <circle r="3" fill={colors.red} opacity="0.8">
            <animateMotion
              dur="2s"
              repeatCount="indefinite"
              path="M100,60 L190,60"
            />
          </circle>
          <circle r="2" fill={colors.red} opacity="0.5">
            <animateMotion
              dur="2s"
              begin="0.7s"
              repeatCount="indefinite"
              path="M100,60 L190,60"
            />
          </circle>
        </>
      )}
      <text
        x="145"
        y="50"
        textAnchor="middle"
        fill={`${colors.red}90`}
        fontSize="7.5"
        fontFamily={fonts.mono}
        opacity={visible ? 1 : 0}
        style={t(0.35)}
      >
        EXPOSED
      </text>
      <rect
        x="196"
        y="20"
        width="158"
        height="80"
        rx="12"
        stroke={colors.red}
        strokeWidth="1.5"
        opacity={visible ? 0.5 : 0}
        style={t(0.4)}
      />
      <text
        x="275"
        y="48"
        textAnchor="middle"
        fill={colors.red}
        fontSize="10"
        fontWeight="600"
        fontFamily={fonts.heading}
        opacity={visible ? 1 : 0}
        style={t(0.45)}
      >
        External AI Servers
      </text>
      <text
        x="275"
        y="64"
        textAnchor="middle"
        fill={`${colors.red}70`}
        fontSize="8"
        fontFamily={fonts.body}
        opacity={visible ? 1 : 0}
        style={t(0.5)}
      >
        OpenAI · Anthropic · etc.
      </text>
      <text
        x="275"
        y="80"
        textAnchor="middle"
        fill={`${colors.red}60`}
        fontSize="7.5"
        fontFamily={fonts.body}
        opacity={visible ? 1 : 0}
        style={t(0.55)}
      >
        Can see your real data
      </text>
      <g opacity={visible ? 1 : 0} style={t(0.5)}>
        <circle cx="148" cy="14" r="10" fill={`${colors.red}20`} />
        <text
          x="148"
          y="18"
          textAnchor="middle"
          fill={colors.red}
          fontSize="13"
          fontWeight="700"
        >
          ⚠
        </text>
      </g>
    </svg>
  );
}

function SafeDiagram({ visible }: { visible: boolean }) {
  const t = (delay: number) => ({ transition: `opacity 0.6s ${delay}s` });
  return (
    <svg
      viewBox="0 0 360 120"
      fill="none"
      style={{
        width: "100%",
        maxWidth: 360,
        height: "auto",
        display: "block",
        margin: "0 auto",
      }}
    >
      <rect
        x="4"
        y="30"
        width="72"
        height="60"
        rx="12"
        stroke={colors.green}
        strokeWidth="1.5"
        opacity={visible ? 0.7 : 0}
        style={t(0.1)}
      />
      <text
        x="40"
        y="56"
        textAnchor="middle"
        fill={colors.green}
        fontSize="9.5"
        fontWeight="600"
        fontFamily={fonts.heading}
        opacity={visible ? 1 : 0}
        style={t(0.2)}
      >
        Your App
      </text>
      <text
        x="40"
        y="70"
        textAnchor="middle"
        fill={`${colors.green}80`}
        fontSize="7.5"
        fontFamily={fonts.body}
        opacity={visible ? 1 : 0}
        style={t(0.25)}
      >
        + PII
      </text>
      <line
        x1="80"
        y1="60"
        x2="118"
        y2="60"
        stroke={colors.green}
        strokeWidth="1.2"
        strokeDasharray="4 3"
        opacity={visible ? 0.4 : 0}
        style={t(0.3)}
      />
      <g opacity={visible ? 1 : 0} style={t(0.35)}>
        <rect
          x="120"
          y="18"
          width="90"
          height="84"
          rx="14"
          fill={`${colors.green}08`}
          stroke={colors.green}
          strokeWidth="1.5"
        />
        <path
          d="M165,30 L148,38 V52 C148,62 155,70 165,74 C175,70 182,62 182,52 V38 Z"
          stroke={colors.green}
          strokeWidth="1.3"
          fill={`${colors.green}15`}
        />
        <path
          d="M158,50 L163,55 L173,45"
          stroke={colors.green}
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <text
          x="165"
          y="88"
          textAnchor="middle"
          fill={colors.green}
          fontSize="9"
          fontWeight="700"
          fontFamily={fonts.heading}
          letterSpacing="0.06em"
        >
          KIJI
        </text>
      </g>
      <line
        x1="214"
        y1="60"
        x2="248"
        y2="60"
        stroke={colors.blue}
        strokeWidth="1.2"
        strokeDasharray="4 3"
        opacity={visible ? 0.4 : 0}
        style={t(0.5)}
      />
      <text
        x="231"
        y="50"
        textAnchor="middle"
        fill={`${colors.blue}90`}
        fontSize="7"
        fontFamily={fonts.mono}
        opacity={visible ? 1 : 0}
        style={t(0.5)}
      >
        MASKED
      </text>
      {visible && (
        <circle r="2.5" fill={colors.green} opacity="0.7">
          <animateMotion
            dur="2.5s"
            repeatCount="indefinite"
            path="M214,60 L248,60"
          />
        </circle>
      )}
      <rect
        x="252"
        y="30"
        width="104"
        height="60"
        rx="12"
        stroke={colors.blue}
        strokeWidth="1.2"
        opacity={visible ? 0.5 : 0}
        style={t(0.55)}
      />
      <text
        x="304"
        y="54"
        textAnchor="middle"
        fill={colors.blue}
        fontSize="9.5"
        fontWeight="600"
        fontFamily={fonts.heading}
        opacity={visible ? 1 : 0}
        style={t(0.6)}
      >
        AI Servers
      </text>
      <text
        x="304"
        y="70"
        textAnchor="middle"
        fill={`${colors.blue}70`}
        fontSize="7.5"
        fontFamily={fonts.body}
        opacity={visible ? 1 : 0}
        style={t(0.65)}
      >
        Only see dummy data
      </text>
    </svg>
  );
}

function DataPipeline({ visible }: { visible: boolean }) {
  const t = (delay: number) => ({ transition: `opacity 0.5s ${delay}s` });
  return (
    <div
      style={{
        padding: "16px 10px",
        borderRadius: 14,
        background: "rgba(255,255,255,0.015)",
        border: "1px solid rgba(255,255,255,0.06)",
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(16px)",
        transition: "all 0.7s cubic-bezier(0.4,0,0.2,1) 0.2s",
        overflow: "hidden",
      }}
    >
      <svg
        viewBox="0 0 440 130"
        fill="none"
        style={{ width: "100%", height: "auto", display: "block" }}
      >
        <g opacity={visible ? 1 : 0} style={t(0.3)}>
          <rect
            x="6"
            y="28"
            width="80"
            height="74"
            rx="12"
            fill={`${colors.blue}08`}
            stroke={colors.blue}
            strokeWidth="1.3"
          />
          <text
            x="46"
            y="18"
            textAnchor="middle"
            fill={colors.blue}
            fontSize="8"
            fontWeight="700"
            fontFamily={fonts.mono}
            letterSpacing="0.06em"
          >
            INPUT
          </text>
          <text
            x="46"
            y="52"
            textAnchor="middle"
            fill={colors.text}
            fontSize="8.5"
            fontWeight="600"
            fontFamily={fonts.heading}
          >
            Your Prompt
          </text>
          <text
            x="46"
            y="66"
            textAnchor="middle"
            fill={`${colors.red}90`}
            fontSize="7"
            fontFamily={fonts.mono}
          >
            john@mail.com
          </text>
          <text
            x="46"
            y="78"
            textAnchor="middle"
            fill={`${colors.red}90`}
            fontSize="7"
            fontFamily={fonts.mono}
          >
            555-012-3456
          </text>
          <text
            x="46"
            y="90"
            textAnchor="middle"
            fill={`${colors.red}90`}
            fontSize="7"
            fontFamily={fonts.mono}
          >
            4242...4242
          </text>
        </g>
        <g opacity={visible ? 1 : 0} style={t(0.45)}>
          <line
            x1="90"
            y1="65"
            x2="130"
            y2="65"
            stroke={colors.blue}
            strokeWidth="1"
            strokeDasharray="4 3"
            opacity="0.4"
          />
          <polygon
            points="128,61 136,65 128,69"
            fill={colors.blue}
            opacity="0.5"
          />
        </g>
        <g opacity={visible ? 1 : 0} style={t(0.5)}>
          <rect
            x="138"
            y="20"
            width="96"
            height="90"
            rx="14"
            fill={`${colors.purple}06`}
            stroke={colors.purple}
            strokeWidth="1.3"
          />
          <text
            x="186"
            y="14"
            textAnchor="middle"
            fill={colors.purple}
            fontSize="8"
            fontWeight="700"
            fontFamily={fonts.mono}
            letterSpacing="0.06em"
          >
            ML SCAN
          </text>
          <circle
            cx="186"
            cy="50"
            r="12"
            stroke={colors.purple}
            strokeWidth="1"
            opacity="0.5"
          />
          <circle
            cx="186"
            cy="50"
            r="6"
            stroke={colors.purple}
            strokeWidth="1"
            opacity="0.7"
          />
          <circle cx="186" cy="50" r="2" fill={colors.purple} opacity="0.8" />
          {visible && (
            <>
              <circle r="1.5" fill={colors.purple} opacity="0.6">
                <animateMotion
                  dur="3s"
                  repeatCount="indefinite"
                  path="M186,38 C198,44 198,56 186,62 C174,56 174,44 186,38"
                />
              </circle>
              <circle r="1" fill={colors.purple} opacity="0.4">
                <animateMotion
                  dur="2.5s"
                  begin="1s"
                  repeatCount="indefinite"
                  path="M174,50 C180,38 192,38 198,50 C192,62 180,62 174,50"
                />
              </circle>
            </>
          )}
          <text
            x="186"
            y="82"
            textAnchor="middle"
            fill={colors.purple}
            fontSize="7.5"
            fontWeight="600"
            fontFamily={fonts.body}
          >
            PII Detected
          </text>
          <text
            x="186"
            y="94"
            textAnchor="middle"
            fill={`${colors.purple}70`}
            fontSize="7"
            fontFamily={fonts.body}
          >
            16+ types
          </text>
        </g>
        <g opacity={visible ? 1 : 0} style={t(0.65)}>
          <line
            x1="238"
            y1="65"
            x2="278"
            y2="65"
            stroke={colors.purple}
            strokeWidth="1"
            strokeDasharray="4 3"
            opacity="0.4"
          />
          <polygon
            points="276,61 284,65 276,69"
            fill={colors.purple}
            opacity="0.5"
          />
        </g>
        <g opacity={visible ? 1 : 0} style={t(0.7)}>
          <rect
            x="286"
            y="28"
            width="80"
            height="74"
            rx="12"
            fill={`${colors.green}08`}
            stroke={colors.green}
            strokeWidth="1.3"
          />
          <text
            x="326"
            y="18"
            textAnchor="middle"
            fill={colors.green}
            fontSize="8"
            fontWeight="700"
            fontFamily={fonts.mono}
            letterSpacing="0.06em"
          >
            OUTPUT
          </text>
          <text
            x="326"
            y="52"
            textAnchor="middle"
            fill={colors.text}
            fontSize="8.5"
            fontWeight="600"
            fontFamily={fonts.heading}
          >
            Safe Prompt
          </text>
          <text
            x="326"
            y="66"
            textAnchor="middle"
            fill={`${colors.green}90`}
            fontSize="7"
            fontFamily={fonts.mono}
          >
            t.roe@web.io
          </text>
          <text
            x="326"
            y="78"
            textAnchor="middle"
            fill={`${colors.green}90`}
            fontSize="7"
            fontFamily={fonts.mono}
          >
            555-999-8765
          </text>
          <text
            x="326"
            y="90"
            textAnchor="middle"
            fill={`${colors.green}90`}
            fontSize="7"
            fontFamily={fonts.mono}
          >
            5111...6677
          </text>
        </g>
        <g opacity={visible ? 1 : 0} style={t(0.8)}>
          <rect
            x="370"
            y="40"
            width="64"
            height="50"
            rx="10"
            fill="transparent"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth="1"
            strokeDasharray="3 3"
          />
          <path
            d="M402,50 L392,55 V63 C392,68 396,72 402,74 C408,72 412,68 412,63 V55 Z"
            stroke={colors.green}
            strokeWidth="1.2"
            fill={`${colors.green}10`}
          />
          <path
            d="M398,61 L401,64 L407,58"
            stroke={colors.green}
            strokeWidth="1.3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <text
            x="402"
            y="84"
            textAnchor="middle"
            fill="rgba(255,255,255,0.25)"
            fontSize="6.5"
            fontFamily={fonts.body}
          >
            On-Device
          </text>
        </g>
      </svg>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   PANE COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

function WhyYaak() {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 150);
    return () => clearTimeout(t);
  }, []);

  return (
    <PaneContainer>
      <Particles />
      <AmbientGlow
        color={`${colors.red}09`}
        size={200}
        position={{ top: "5%", left: "20%" }}
      />
      <AmbientGlow
        color={`${colors.green}08`}
        size={250}
        position={{ bottom: "10%", right: "15%" }}
      />

      <PaneTitle
        subtitle="AI services receive your raw prompts — including names, emails, and sensitive data."
        visible={visible}
      >
        Why{" "}
        <GradientText gradient={gradients.blueToPurple}>
          Privacy Proxy
        </GradientText>
        ?
      </PaneTitle>

      <div style={{ marginBottom: 6 }}>
        <SectionHeader
          color={colors.red}
          label="Without Yaak"
          visible={visible}
          delay={0.15}
        />
        <div
          style={{
            padding: "14px 12px",
            borderRadius: 12,
            background: `${colors.red}08`,
            border: `1px solid ${colors.red}1A`,
            opacity: visible ? 1 : 0,
            transform: visible ? "translateY(0)" : "translateY(12px)",
            transition: `${transitions.default} 0.2s`,
          }}
        >
          <DangerDiagram visible={visible} />
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 10,
          margin: "10px 0",
          opacity: visible ? 1 : 0,
          transition: "opacity 0.5s 0.45s",
        }}
      >
        <div
          style={{
            flex: 1,
            height: 1,
            background:
              "linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent)",
          }}
        />
        <span
          style={{
            fontSize: 9,
            fontWeight: 700,
            color: "rgba(255,255,255,0.2)",
            letterSpacing: "0.12em",
            fontFamily: fonts.mono,
          }}
        >
          VS
        </span>
        <div
          style={{
            flex: 1,
            height: 1,
            background:
              "linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent)",
          }}
        />
      </div>

      <div style={{ marginBottom: 16 }}>
        <SectionHeader
          color={colors.green}
          label="With Yaak"
          visible={visible}
          delay={0.5}
        />
        <div
          style={{
            padding: "14px 12px",
            borderRadius: 12,
            background: `${colors.green}05`,
            border: `1px solid ${colors.green}1A`,
            opacity: visible ? 1 : 0,
            transform: visible ? "translateY(0)" : "translateY(12px)",
            transition: `${transitions.default} 0.55s`,
          }}
        >
          <SafeDiagram visible={visible} />
        </div>
      </div>

      <div style={{ marginBottom: 16 }}>
        <div
          style={{
            fontSize: 9,
            fontWeight: 600,
            color: "rgba(180,190,210,0.35)",
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            marginBottom: 6,
            textAlign: "center",
            opacity: visible ? 1 : 0,
            transition: "opacity 0.5s 0.75s",
          }}
        >
          Real-time PII masking examples
        </div>
        <PIITicker visible={visible} />
      </div>

      <div
        style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}
      >
        <StatBadge
          icon="🔒"
          value="100%"
          label="On-device"
          color={colors.green}
          delay={0.85}
          visible={visible}
        />
        <StatBadge
          icon="🤖"
          value="Local ML"
          label="PII detection"
          color={colors.purple}
          delay={0.95}
          visible={visible}
        />
        <StatBadge
          icon="🚫"
          value="Zero"
          label="3rd party sharing"
          color={colors.orange}
          delay={1.05}
          visible={visible}
        />
      </div>

      <TrustBadge
        text="Your data never leaves your device unprotected"
        visible={visible}
        delay={1.15}
      />
    </PaneContainer>
  );
}

function YaakWorkflow() {
  const [animated, setAnimated] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 100);
    return () => clearTimeout(t);
  }, []);

  return (
    <PaneContainer>
      <AmbientGlow
        color={`${colors.blue}0A`}
        size={200}
        position={{ top: "10%", right: "15%" }}
      />
      <AmbientGlow
        color={`${colors.purple}08`}
        size={150}
        position={{ bottom: "5%", left: "10%" }}
      />

      <PaneTitle
        subtitle="Yaak acts as a transparent proxy between your app and AI services."
        visible={animated}
      >
        <GradientText>How It Works</GradientText>
      </PaneTitle>

      <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
        {workflowSteps.map((step, i) => (
          <div key={i}>
            <StepCard step={step} index={i} animated={animated} />
            {i < workflowSteps.length - 1 && (
              <ArrowConnector
                color={step.color}
                nextColor={workflowSteps[i + 1].color}
                animated={animated}
                index={i}
              />
            )}
          </div>
        ))}
      </div>

      <div style={{ marginTop: 16 }}>
        <TrustBadge
          text="100% local processing · Zero data leakage"
          visible={animated}
          delay={1.1}
          size={14}
        />
      </div>
    </PaneContainer>
  );
}

function WhatHappensToYourData() {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 150);
    return () => clearTimeout(t);
  }, []);

  return (
    <PaneContainer>
      <AmbientGlow
        color={`${colors.purple}09`}
        size={200}
        position={{ top: "8%", right: "20%" }}
      />
      <AmbientGlow
        color={`${colors.green}08`}
        size={180}
        position={{ bottom: "8%", left: "12%" }}
      />

      <PaneTitle
        subtitle="On-device ML identifies and masks PII before it ever leaves your device."
        visible={visible}
      >
        What Happens to <GradientText>Your Data</GradientText>?
      </PaneTitle>

      <DataPipeline visible={visible} />

      <div style={{ marginTop: 18, marginBottom: 16 }}>
        <SectionHeader
          color={colors.green}
          label="Our Privacy Promise"
          visible={visible}
          delay={0.8}
        />
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {privacyPromises.map((p, i) => (
            <PromiseCard key={i} item={p} index={i} visible={visible} />
          ))}
        </div>
      </div>

      <TrustBadge
        text="Open source · Verifiable · Privacy-first"
        visible={visible}
        delay={1.5}
      />
    </PaneContainer>
  );
}

function PIIEntities() {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 150);
    return () => clearTimeout(t);
  }, []);

  return (
    <PaneContainer>
      <Particles />
      <AmbientGlow
        color={`${colors.blue}0A`}
        size={220}
        position={{ top: "5%", left: "15%" }}
      />
      <AmbientGlow
        color={`${colors.purple}08`}
        size={200}
        position={{ bottom: "10%", right: "20%" }}
      />
      <AmbientGlow
        color={`${colors.orange}06`}
        size={150}
        position={{ top: "50%", right: "5%" }}
      />

      <PaneTitle
        subtitle="Kiji automatically detects and masks a wide range of sensitive information."
        visible={visible}
      >
        <GradientText gradient={gradients.blueToPurple}>
          PII Entities
        </GradientText>{" "}
        We Detect
      </PaneTitle>

      <div style={{ marginBottom: 20 }}>
        <SectionHeader
          color={colors.purple}
          label="16+ PII Types Detected"
          visible={visible}
          delay={0.2}
        />
        <PIIGrid visible={visible} />
      </div>

      <div
        style={{
          padding: "16px",
          borderRadius: 12,
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)",
          opacity: visible ? 1 : 0,
          transform: visible ? "translateY(0)" : "translateY(12px)",
          transition: `${transitions.default} 1.2s`,
          marginBottom: 16,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 12,
          }}
        >
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: `${colors.green}10`,
              border: `1px solid ${colors.green}25`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <svg width="18" height="18" viewBox="0 0 20 20" fill="none">
              <path
                d="M10 1L3 4.5V9.5C3 14.2 6 17.5 10 19C14 17.5 17 14.2 17 9.5V4.5L10 1Z"
                stroke={colors.green}
                strokeWidth="1.5"
                strokeLinejoin="round"
              />
              <path
                d="M7 10l2 2 4-4"
                stroke={colors.green}
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: colors.text,
                fontFamily: fonts.heading,
                marginBottom: 2,
              }}
            >
              Smart Detection
            </div>
            <div
              style={{
                fontSize: 11,
                color: colors.textMuted,
                fontFamily: fonts.body,
              }}
            >
              Our ML model continuously improves to catch new PII patterns
            </div>
          </div>
        </div>
        <div
          style={{
            display: "flex",
            gap: 8,
            flexWrap: "wrap",
          }}
        >
          {["Context-aware", "Low false positives", "Customizable"].map(
            (tag, i) => (
              <span
                key={i}
                style={{
                  fontSize: 9,
                  fontWeight: 500,
                  color: `${colors.green}90`,
                  fontFamily: fonts.mono,
                  background: `${colors.green}10`,
                  padding: "4px 8px",
                  borderRadius: 6,
                  border: `1px solid ${colors.green}20`,
                }}
              >
                {tag}
              </span>
            )
          )}
        </div>
      </div>

      <TrustBadge
        text="All detection happens locally on your device"
        visible={visible}
        delay={1.4}
      />
    </PaneContainer>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN MODAL COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */

interface WelcomeModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function WelcomeModal({ isOpen, onClose }: WelcomeModalProps) {
  const [dontShowAgain, setDontShowAgain] = useState(false);
  const [currentPane, setCurrentPane] = useState(0);

  const isElectron =
    typeof window !== "undefined" && window.electronAPI !== undefined;

  if (!isOpen) return null;

  const handleClose = async () => {
    if (dontShowAgain && isElectron && window.electronAPI) {
      try {
        await window.electronAPI.setWelcomeDismissed(true);
      } catch (error) {
        console.error("Failed to save welcome dismissed preference:", error);
      }
    }
    onClose();
  };

  const totalPanes = 4;
  const isFirstPane = currentPane === 0;
  const isLastPane = currentPane === totalPanes - 1;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full flex flex-col">
        <div className="flex items-center justify-between p-6 border-b border-slate-200 flex-shrink-0">
          <div className="flex items-center gap-3">
            <Shield className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-slate-800">
              Welcome to Kiji Privacy Proxy
            </h2>
          </div>
          <button
            onClick={handleClose}
            className="p-1 text-slate-400 hover:text-slate-600 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6">
          <div style={{ position: "relative" }}>
            {/* Invisible spacer: always rendered to lock container height to tallest pane */}
            <div aria-hidden style={{ visibility: "hidden" }}>
              <WhyYaak />
            </div>
            {/* Active pane rendered absolutely on top */}
            <div style={{ position: "absolute", top: 0, left: 0, right: 0 }}>
              {currentPane === 0 && <WhyYaak />}
              {currentPane === 1 && <YaakWorkflow />}
              {currentPane === 2 && <WhatHappensToYourData />}
              {currentPane === 3 && <PIIEntities />}
            </div>
          </div>
        </div>

        <div className="p-6 pt-0 flex-shrink-0 space-y-4">
          <div className="flex justify-center gap-2">
            {Array.from({ length: totalPanes }).map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentPane(index)}
                className={`w-2 h-2 rounded-full transition-colors ${
                  index === currentPane
                    ? "bg-blue-600"
                    : "bg-slate-300 hover:bg-slate-400"
                }`}
                aria-label={`Go to page ${index + 1}`}
              />
            ))}
          </div>

          {isLastPane && (
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={dontShowAgain}
                onChange={(e) => setDontShowAgain(e.target.checked)}
                className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
              />
              <span className="text-sm text-slate-600">
                Don't show this again
              </span>
            </label>
          )}

          <div className="flex gap-3">
            {!isFirstPane && (
              <button
                onClick={() => setCurrentPane(currentPane - 1)}
                className="flex-1 px-4 py-3 border border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors font-medium flex items-center justify-center gap-2"
              >
                <ChevronLeft className="w-4 h-4" />
                Back
              </button>
            )}
            {isLastPane ? (
              <button
                onClick={handleClose}
                className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
              >
                Get Started
              </button>
            ) : (
              <button
                onClick={() => setCurrentPane(currentPane + 1)}
                className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium flex items-center justify-center gap-2"
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
