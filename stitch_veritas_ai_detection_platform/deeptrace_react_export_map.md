# DeepTrace React Application Architecture & Export Map

This document outlines the structure, theme logic, and component organization for the DeepTrace React application, incorporating the latest synchronized designs for both Light and Dark modes across Desktop and Mobile.

## 1. Project Structure
```text
/src
  /assets          # Brand logos, icons, and static forensic media
  /components      # Reusable UI components (Atomic Design)
    /ui            # Base components (Buttons, Inputs, Cards - Skeuomorphic/Tactile)
    /layout        # Navigation, Footers, Page Shells
    /forensic      # Domain-specific components (Upload, Neural Feed, Analysis)
  /hooks           # Theme and interaction hooks
  /pages           # Main page views
    /landing       # DeepTrace Home (Skeuomorphic Landing)
    /docs          # API & Webhooks Documentation
  /styles          # Tailwind config, global CSS, and Design System tokens
  /store           # Theme state management (Zustand/Context API)
  /utils           # Code syntax highlighting and formatting helpers
```

## 2. Global Theme Management
The application utilizes a `ThemeContext` to manage the switch between **Nocturne Tactile (Dark)** and **Tactile Monochrome (Light)**.
- **Logic:** Toggles a `dark` class on the root element.
- **Tokens:** CSS Variables mapped to Design Systems {{DATA:DESIGN_SYSTEM:DESIGN_SYSTEM_1}} and {{DATA:DESIGN_SYSTEM:DESIGN_SYSTEM_2}}.
- **Transitions:** Integrated smooth CSS transitions for surfaces and typography during the theme swap.

## 3. Screen Mapping (Latest Versions)
The export uses the following finalized source screens:
- **Desktop Dark:** {{DATA:SCREEN:SCREEN_9}}
- **Desktop Light:** {{DATA:SCREEN:SCREEN_26}}
- **Mobile Dark:** {{DATA:SCREEN:SCREEN_21}}
- **Mobile Light:** {{DATA:SCREEN:SCREEN_29}}
- **Documentation:** {{DATA:SCREEN:SCREEN_32}} (Unified API & Webhooks)

## 4. Key Implementation Details
- **Branding:** "DeepTrace" typography with custom tracking and weight pairings.
- **Navigation:** Integrated GitHub button linking to `github.com/obstinix` and tactile theme toggle.
- **Documentation:** High-density terminal aesthetics with recessed panels and responsive code blocks.
- **Responsiveness:** Tailwind-driven grid systems and flexible containers for seamless scaling from mobile to ultra-wide desktop.
- **Interactivity:** Micro-animations for "Secure Access" buttons, "Copy Code" triggers, and skeuomorphic hover states.

---
*Ready for handoff to backend and ML integration teams.*
