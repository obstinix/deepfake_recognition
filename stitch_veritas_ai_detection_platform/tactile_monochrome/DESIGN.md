---
name: Tactile Monochrome
colors:
  surface: '#fbf9f9'
  surface-dim: '#dbdad9'
  surface-bright: '#fbf9f9'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f5f3f3'
  surface-container: '#efeded'
  surface-container-high: '#e9e8e7'
  surface-container-highest: '#e3e2e2'
  on-surface: '#1b1c1c'
  on-surface-variant: '#4c4546'
  inverse-surface: '#303031'
  inverse-on-surface: '#f2f0f0'
  outline: '#7e7576'
  outline-variant: '#cfc4c5'
  surface-tint: '#5e5e5e'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#1b1b1b'
  on-primary-container: '#848484'
  inverse-primary: '#c6c6c6'
  secondary: '#5d5f5f'
  on-secondary: '#ffffff'
  secondary-container: '#dfe0e0'
  on-secondary-container: '#616363'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#1b1b1b'
  on-tertiary-container: '#848484'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e2e2e2'
  primary-fixed-dim: '#c6c6c6'
  on-primary-fixed: '#1b1b1b'
  on-primary-fixed-variant: '#474747'
  secondary-fixed: '#e2e2e2'
  secondary-fixed-dim: '#c6c6c7'
  on-secondary-fixed: '#1a1c1c'
  on-secondary-fixed-variant: '#454747'
  tertiary-fixed: '#e2e2e2'
  tertiary-fixed-dim: '#c6c6c6'
  on-tertiary-fixed: '#1b1b1b'
  on-tertiary-fixed-variant: '#474747'
  background: '#fbf9f9'
  on-background: '#1b1c1c'
  surface-variant: '#e3e2e2'
typography:
  display:
    fontFamily: Plus Jakarta Sans
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.2'
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  label-mono:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: '1.0'
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 8px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 64px
  max-width: 1280px
---

## Brand & Style

This design system is defined by a rigorous adherence to grayscale skeuomorphism, blending the tactile physicality of analog interfaces with a modern, cinematic digital aesthetic. The intent is to evoke a sense of permanence, precision, and high-end hardware. 

The visual language relies on simulated depth rather than color to communicate hierarchy. By utilizing soft shadows, precise highlights, and "pressed" vs "raised" states, the interface provides immediate intuitive feedback through perceived physicality. The atmosphere is professional, focused, and quiet, allowing the content to remain the sole focus within a sophisticated, structural frame.

## Colors

The palette is strictly achromatic. Vibrancy is achieved through contrast and lighting effects rather than hue. 

- **Light Mode:** Uses a mid-light gray base (`#E0E0E0`) which allows for both white highlights and dark shadows to remain visible. Primary actions and surfaces are rendered in a slightly darker "inset" or "extruded" gray to provide depth.
- **Dark Mode:** Utilizes a deep charcoal base (`#1A1A1A`). Highlights move toward a subtle light-gray, while shadows sink into pure black.
- **Functional Grays:** A spectrum of grays provides clear differentiation for borders, disabled states, and secondary text without breaking the monochrome constraint.

## Typography

The typography strategy balances the organic curves of Plus Jakarta Sans with the technical precision of JetBrains Mono. 

- **Headlines:** Use Plus Jakarta Sans with tight tracking and bold weights to create a "cinematic" and commanding presence.
- **Body:** Plus Jakarta Sans provides high readability with a contemporary, friendly feel that softens the industrial nature of the grayscale palette.
- **Labels & Metadata:** JetBrains Mono is used for small-scale technical data, labels, and status indicators. This reinforces the "instrumental" feel of the skeuomorphic elements.

## Layout & Spacing

The layout follows a structured 12-column fixed grid for desktop, transitioning to a fluid 4-column system for mobile. Because skeuomorphic elements require "room to breathe" to prevent shadow overlap and visual clutter, the spacing system is generous.

Alignment is strictly mathematical, emphasizing the "machined" aesthetic. Margin and padding should always be multiples of the 8px base unit. Component containers use internal padding that scales with the perceived elevation: higher "raised" elements require more surrounding negative space to justify their visual weight.

## Elevation & Depth

Hierarchy is established through light physics. Surfaces do not simply sit on top of each other; they are carved into or extruded from the background.

- **Raised State:** Uses two shadows. A light shadow (highlight) on the top-left and a dark shadow on the bottom-right. This creates the illusion of the element pushing out toward the user.
- **Pressed/Inset State:** Uses two inner shadows. A dark shadow on the top-left and a light shadow (highlight) on the bottom-right. This simulates the element being physically pushed into the surface.
- **Surface Fill:** Main components (buttons, cards) should have a slightly darker fill than the global background. This "hollows out" the component, giving it a heavier, more substantial presence.
- **Bevels:** A 1px subtle border can be used to simulate a "milled" edge, especially on high-priority interactive components.

## Shapes

The design system utilizes a "Soft-Industrial" shape language. While the underlying grid is rigid, the corners are rounded to facilitate the soft shadow gradients required for skeuomorphism. 

Standard components use a 0.5rem (8px) radius. Larger cards or containers use 1rem (16px). This roundedness is essential for the "squishy" tactile feel; sharp corners tend to break the illusion of physical light casting and make the shadows appear artificial.

## Components

- **Buttons:** Rendered with a darker fill than the background. In the default state, they are "Raised." On hover, the elevation increases (larger shadow spread). On click, they transition to "Inset" (inner shadows) to mimic a physical depress.
- **Inputs:** Always "Inset" (carved into the surface). Use JetBrains Mono for input text to emphasize the data-entry feel. The "well" of the input should be slightly darker than the surface.
- **Cards:** Subtle "Raised" elevation. Unlike buttons, cards do not have a darker fill; they match the background but are defined by their soft perimeter shadows.
- **Chips/Toggles:** Use the "Inset" well as a track and a "Raised" circle as the handle. This provides a clear mental model of a sliding physical switch.
- **Lists:** Separated by "Etched" lines—a 1px dark line with a 1px light highlight immediately below it to create a recessed groove effect.
- **Checkboxes:** Small "Inset" squares that reveal a high-contrast white or black mark when active.