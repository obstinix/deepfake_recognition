# Deepfake Recognition Frontend

This is the modern, cinematic frontend for the Deepfake Recognition system, built with React, TypeScript, and Vite.

## Features

- **Cinematic UI**: Modern dark theme with animated gradients, glassmorphism, and smooth micro-interactions.
- **Drag & Drop Uploads**: Seamless file upload experience for images and videos.
- **Real-time Polling**: Asynchronous task tracking with live status updates.
- **Explainable AI Results**: Interactive Grad-CAM heatmap viewer and ensemble confidence gauges.
- **Responsive Design**: Fully mobile-optimized interface using Tailwind CSS.

## Stack

- **React 18**
- **TypeScript**
- **Vite** (Build tool & Dev server)
- **Tailwind CSS** (Styling)
- **React Router** (Navigation)
- **Axios** (API requests)

## Setup & Running

It is highly recommended to run the frontend via Docker Compose from the root directory:

```bash
cd ..
docker-compose up --build frontend
```

### Running Locally (Without Docker)

1. Install dependencies:
   ```bash
   npm install
   ```
2. Run the development server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:3000` in your browser.

## Build for Production

```bash
npm run build
```
The output will be generated in the `dist/` directory, ready to be served by Nginx.
