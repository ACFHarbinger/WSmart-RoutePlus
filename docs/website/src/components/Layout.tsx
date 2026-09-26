import { NavLink, Outlet } from "react-router-dom";
import { githubRepo, navLinks } from "../data/site";
import { useState, useEffect } from "react";

type Theme = "light" | "dark";

function initialTheme(): Theme {
  const stored = window.localStorage.getItem("wsmart-theme");
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function Layout() {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("wsmart-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(t => t === "light" ? "dark" : "light");
  };

  return (
    <div className="site-shell">
      <div className="atmosphere" aria-hidden="true" />

      <header className="site-nav">
        <div className="nav-container">
          <NavLink to="/" className="brand" end style={{ display: 'flex', alignItems: 'center', gap: '12px', fontWeight: 600 }}>
            <span>WSmart+ Route</span>
          </NavLink>

          <nav className="nav-links" aria-label="Primary">
            {navLinks.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
              >
                {link.label}
              </NavLink>
            ))}
          </nav>

          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <button onClick={toggleTheme} aria-label="Toggle theme" style={{ fontSize: '0.875rem' }}>
              {theme === 'light' ? 'Dark Mode' : 'Light Mode'}
            </button>
            <a
              className="nav-link"
              href={githubRepo}
              target="_blank"
              rel="noreferrer"
            >
              GitHub
            </a>
          </div>
        </div>
      </header>

      <main className="main-content">
        <Outlet />
      </main>

      <footer style={{ borderTop: '1px solid var(--border-subtle)', padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem', fontFamily: 'var(--font-mono)' }}>
        <span>WSmart+ Route · combinatorial optimization for waste collection</span>
        <br />
        <span>Research platform · Studio desktop · open methods</span>
      </footer>
    </div>
  );
}
