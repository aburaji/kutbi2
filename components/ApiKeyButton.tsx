import React from 'react';

interface ApiKeyButtonProps {
    onOpen: () => void;
}

const ApiKeyButton: React.FC<ApiKeyButtonProps> = ({ onOpen }) => {
    return (
        <button
            onClick={onOpen}
            className="fixed bottom-6 right-6 bg-blue-600 text-white rounded-full p-4 shadow-lg hover:bg-blue-700 transition-all duration-300 ease-in-out z-40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
            aria-label="إضافة أو تعديل مفتاح API"
            data-tooltip="إضافة/تعديل مفتاح API"
        >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 7h2a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2V9a2 2 0 012-2h2m4 0h2M9 12l2 2 4-4m-6 8h.01" />
            </svg>
        </button>
    );
};

export default ApiKeyButton;
