import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Benchmarks from "./pages/Benchmarks";
import Docs from "./pages/Docs";
import Home from "./pages/Home";
import Platform from "./pages/Platform";
import Research from "./pages/Research";
import Roadmap from "./pages/Roadmap";
import Simulation from "./pages/Simulation";
import Studio from "./pages/Studio";
import "./styles/site.css";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="platform" element={<Platform />} />
          <Route path="research" element={<Research />} />
          <Route path="benchmarks" element={<Benchmarks />} />
          <Route path="simulation" element={<Simulation />} />
          <Route path="studio" element={<Studio />} />
          <Route path="docs" element={<Docs />} />
          <Route path="roadmap" element={<Roadmap />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
