---
name: Nocturne Tactile
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#393939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#201f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353534'
  on-surface: '#e5e2e1'
  on-surface-variant: '#c4c7c8'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#8e9192'
  outline-variant: '#444748'
  surface-tint: '#c6c6c7'
  primary: '#ffffff'
  on-primary: '#2f3131'
  primary-container: '#e2e2e2'
  on-primary-container: '#636565'
  inverse-primary: '#5d5f5f'
  secondary: '#c7c6c6'
  on-secondary: '#303031'
  secondary-container: '#464747'
  on-secondary-container: '#b5b5b5'
  tertiary: '#ffffff'
  on-tertiary: '#2f3131'
  tertiary-container: '#e2e2e2'
  on-tertiary-container: '#636565'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e2e2e2'
  primary-fixed-dim: '#c6c6c7'
  on-primary-fixed: '#1a1c1c'
  on-primary-fixed-variant: '#454747'
  secondary-fixed: '#e3e2e2'
  secondary-fixed-dim: '#c7c6c6'
  on-secondary-fixed: '#1b1c1c'
  on-secondary-fixed-variant: '#464747'
  tertiary-fixed: '#e2e2e2'
  tertiary-fixed-dim: '#c6c6c7'
  on-tertiary-fixed: '#1a1c1c'
  on-tertiary-fixed-variant: '#454747'
  background: '#131313'
  on-background: '#e5e2e1'
  surface-variant: '#353534'
typography:
  display:
    fontFamily: Plus Jakarta Sans
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  data-lg:
    fontFamily: JetBrains Mono
    fontSize: 16px
    fontWeight: '500'
    lineHeight: 24px
  data-sm:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 48px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 64px
---

## Brand & Style

This design system is a sophisticated, research-grade interface that prioritizes physical intuition and focus. The brand personality is quiet, authoritative, and precise, designed for high-stakes environments where clarity and technical rigor are paramount. 

The aesthetic centers on **Tactile Skeuomorphism** within a strictly monochrome palette. It moves away from flat "digital-first" trends toward a UI that feels machined from physical material. Every element should feel like a tangible object with weight, depth, and a matte finish. By using deep matte charcoal as the primary medium, the design system evokes a sense of "dark laboratory" or "high-end instrumentation," reducing eye strain while maximizing the perceived value of the data being presented.

The emotional response should be one of calm, professional confidence. There are no decorative flourishes; every shadow and highlight serves to define the physical boundaries of a control or a container.

## Colors

The color strategy is strictly grayscale, relying entirely on luminance and shadow to create hierarchy. 

- **Base Background:** Deep matte charcoal (#121212) serves as the "floor" of the interface.
- **Surface (Elevated):** A slightly lighter charcoal (#1A1A1A) is used for cards and panels that sit above the base.
- **Recessed (Inverted):** A darker, near-black (#0A0A0A) is used for inputs and depressed buttons to create a "machined-out" look.
- **Typography:** Pure white (#FFFFFF) is reserved for primary headings and critical data. Mid-range grays (#888888 to #CCCCCC) are used for labels and secondary information to maintain a balanced visual weight.

Avoid all saturated colors. Use variations in opacity and lightness to denote state changes (hover, active, disabled) rather than hue shifts.

## Typography

The typography system pairs the modern, approachable clarity of **Plus Jakarta Sans** with the technical precision of **JetBrains Mono**.

**Plus Jakarta Sans** is used for all editorial and structural content. Its high x-height ensures legibility against dark backgrounds. Headlines should use tighter letter-spacing to feel "heavy" and impactful.

**JetBrains Mono** is reserved for "Technical Data"—numerical values, status codes, timestamps, and labels. This reinforces the research-grade nature of the interface, suggesting that the information is being pulled directly from a precise instrument.

All text should favor high-contrast white or light gray. Never use pure black text.

## Layout & Spacing

The layout follows a **Fixed Grid** philosophy on desktop to maintain the feel of a physical control console. 

- **Grid:** Use a 12-column grid for desktop (max-width: 1440px) with 24px gutters.
- **Rhythm:** All spacing is based on a 4px baseline. Components should use 16px (md) or 24px (lg) padding to ensure the tactile "edges" have room to breathe.
- **Responsive Behavior:** On mobile, the grid collapses to a single column with 16px side margins. Cards should span the full width to maximize the tactile surface area.

Layouts should be symmetrical and structured. Use generous margins between "physical" modules to prevent the interface from feeling cluttered.

## Elevation & Depth

Depth is the primary communicator of function in this design system. 

**Elevated Surfaces (Cards):** Use a combination of a light top-edge highlight (1px, white, 10% opacity) and a soft, diffused bottom-right shadow (12px blur, #000000 at 40% opacity). This creates a "matte plastic" look that rises off the background.

**Recessed Surfaces (Inputs/Active States):** Use inner shadows. A top-left inner shadow (4px, #000000 at 50% opacity) and a bottom-right inner highlight (1px, white, 5% opacity). This makes the element look carved into the surface.

**Interaction Layers:** 
- **Idle:** Subtle outer elevation.
- **Hover:** Slightly increased elevation (increased shadow blur).
- **Pressed:** Transition from outer elevation to inner recession (the button physically "sinks").

## Shapes

The shape language is "Soft" (0.25rem - 0.75rem) to mimic industrially molded components. 

- **Standard Elements:** Use `rounded` (4px) for small controls like checkboxes and tiny buttons.
- **Containers:** Use `rounded-lg` (8px) for cards and main UI panels.
- **Inner Elements:** When an element is nested inside a container, its corner radius should be 4px smaller than the container to maintain visual concentricity.

Avoid "Pill" shapes entirely. Rectilinear forms with soft corners convey more professional stability.

## Components

**Buttons:** Buttons are darker than the surface they sit on. They should appear "heavy" and weighted. Use a 1px solid border (#2A2A2A) to define the edge. On click, they must use an inner shadow to appear physically depressed.

**Input Fields:** Always recessed. Use the `recessed` depth style with JetBrains Mono for the input text. The cursor should be a solid white block.

**Cards:** These are the primary containers. They use the `elevated` depth style. Content inside should be padded by at least 24px.

**Chips/Tags:** Monospace text only. Use a subtle 1px border (#333333) and no background fill.

**Data Readouts:** Use JetBrains Mono in a slightly larger weight. Pair with a "Label Caps" style above the data to identify the metric.

**Checkboxes:** Square with a 2px radius. When checked, the box becomes recessed and displays a simple 1px white checkmark. No color fills.