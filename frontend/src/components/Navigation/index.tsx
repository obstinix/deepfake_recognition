import { Link, useLocation } from 'react-router-dom';

export default function Navigation() {
  const location = useLocation();

  const links = [
    { to: '/', label: 'Home' },
    { to: '/detect', label: 'Detect' },
    { to: '/dashboard', label: 'Dashboard' },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-dark-900/80 backdrop-blur-xl border-b border-dark-400/20">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-lg group-hover:shadow-primary-500/30 transition-all duration-300">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <span className="text-xl font-bold text-white tracking-tight">
            Deep<span className="text-primary-400">Detect</span>
          </span>
        </Link>

        <div className="flex items-center gap-1">
          {links.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                location.pathname === link.to
                  ? 'bg-primary-600/20 text-primary-400'
                  : 'text-dark-100 hover:text-white hover:bg-dark-500/50'
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
