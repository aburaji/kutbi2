import React from 'react';

type ActiveTab = 'books' | 'researches' | 'periodicals' | 'videos' | 'audios' | 'images' | 'websites' | 'about' | 'deployment';

interface TabsProps {
    activeTab: ActiveTab;
    onTabChange: (tab: ActiveTab) => void;
}

const BookIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
    </svg>
);

const ResearchIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
        <path fillRule="evenodd" d="M4 5a2 2 0 012-2h8a2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h4a1 1 0 100-2H7zm0 4a1 1 0 100 2h4a1 1 0 100-2H7z" clipRule="evenodd" />
    </svg>
);

const PeriodicalsIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path d="M7 3a1 1 0 000 2h6a1 1 0 100-2H7zM4 7a1 1 0 011-1h10a1 1 0 110 2H5a1 1 0 01-1-1zM2 11a2 2 0 012-2h12a2 2 0 012 2v4a2 2 0 01-2 2H4a2 2 0 01-2-2v-4z" />
    </svg>
);


const VideoIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 001.553.832l3-2a1 1 0 000-1.664l-3-2z" />
    </svg>
);

const AudioIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 1.343-3 3s1.343 3 3 3 3-1.343 3-3V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 1.343-3 3s1.343 3 3 3 3-1.343 3-3V3z" />
    </svg>
);

const ImageIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
    </svg>
);

const WebsiteIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM4.332 8.027a6.012 6.012 0 011.912-2.706C6.512 5.73 6.512 5.73 6.512 5.73a3.5 3.5 0 014.975 0 3.5 3.5 0 014.975 0c.345.345.672.71.961 1.095a6.002 6.002 0 01-1.196 6.464c-.64.64-1.373 1.157-2.182 1.545a3.5 3.5 0 01-4.975 0 3.5 3.5 0 01-4.975 0 6.002 6.002 0 01-1.196-6.464.63.63 0 00-.011-.011zM10 12a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
    </svg>
);

const DeploymentIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
      <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M12 12v9" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 15l-3-3-3 3" />
    </svg>
);

const AboutIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" />
    </svg>
);


const Tabs: React.FC<TabsProps> = ({ activeTab, onTabChange }) => {
    const tabs = [
        { id: 'books', label: 'الكتب', icon: <BookIcon /> },
        { id: 'researches', label: 'البحوث', icon: <ResearchIcon /> },
        { id: 'periodicals', label: 'الدوريات', icon: <PeriodicalsIcon /> },
        { id: 'videos', label: 'الفيديوهات', icon: <VideoIcon /> },
        { id: 'audios', label: 'الصوتيات', icon: <AudioIcon /> },
        { id: 'images', label: 'الصور', icon: <ImageIcon /> },
        { id: 'websites', label: 'مواقع', icon: <WebsiteIcon /> },
        { id: 'deployment', label: 'الجاهزية للنشر', icon: <DeploymentIcon /> },
        { id: 'about', label: 'من نحن', icon: <AboutIcon /> },
    ];

    return (
        <div className="mb-6">
            <div className="sm:hidden">
                <label htmlFor="tabs" className="sr-only">اختر تبويب</label>
                <select
                    id="tabs"
                    name="tabs"
                    className="block w-full rounded-md border-slate-300 py-2 pl-3 pr-10 text-base focus:border-blue-500 focus:outline-none focus:ring-blue-500"
                    value={activeTab}
                    onChange={(e) => onTabChange(e.target.value as ActiveTab)}
                >
                    {tabs.map((tab) => (
                        <option key={tab.id} value={tab.id}>{tab.label}</option>
                    ))}
                </select>
            </div>
            <div className="hidden sm:block">
                <div className="border-b border-slate-200">
                    <nav className="-mb-px flex space-x-4" aria-label="Tabs">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => onTabChange(tab.id as ActiveTab)}
                                className={`
                                    ${activeTab === tab.id
                                        ? 'border-blue-500 text-blue-600'
                                        : 'border-transparent text-slate-500 hover:border-slate-300 hover:text-slate-700'
                                    }
                                    whitespace-nowrap flex py-4 px-1 border-b-2 font-medium text-sm transition-colors
                                `}
                                aria-current={activeTab === tab.id ? 'page' : undefined}
                            >
                                {tab.icon}
                                {tab.label}
                            </button>
                        ))}
                    </nav>
                </div>
            </div>
        </div>
    );
};

export default Tabs;