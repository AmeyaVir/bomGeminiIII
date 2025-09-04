import { useLocation, Link } from 'react-router-dom';
import { BarChart, Home, Upload, Database, Settings, Globe } from 'lucide-react';
import { useTranslation } from '../contexts/TranslationContext';
import LanguageSelector from './LanguageSelector';

function Layout({ children }) {
  const location = useLocation();
  const { t } = useTranslation();

  const navigation = [
    { name: t('navigation.dashboard'), href: '/dashboard', icon: Home },
    { name: t('navigation.upload'), href: '/upload', icon: Upload },
    { name: t('knowledgeBase.title'), href: '/knowledge-base', icon: Database },
    { name: t('settings.title'), href: '/settings', icon: Settings }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <BarChart className="h-8 w-8 text-primary-600" />
                <span className="ml-2 text-xl font-bold text-gray-900">BOM Platform</span>
                <span className="ml-2 text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                  Enhanced v2.0
                </span>
              </div>
              <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname === item.href;
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      className={`${
                        isActive
                          ? 'border-primary-500 text-primary-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      } inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium`}
                    >
                      <Icon className="h-4 w-4 mr-2" />
                      {item.name}
                    </Link>
                  );
                })}
              </div>
            </div>
            <div className="flex items-center">
              <LanguageSelector className="mr-4" />
              <div className="hidden sm:flex items-center space-x-4">
                <span className="text-xs text-gray-500">
                  QA Classification • Knowledge Base • Japanese Support
                </span>
              </div>
            </div>
          </div>
        </div>
      </nav>
      <main className="min-h-screen">
        {children}
      </main>
    </div>
  );
}

export default Layout;
