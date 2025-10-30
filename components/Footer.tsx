import React from 'react';

// Using a reliable thumbnail link for embedding.
const bannerImageUrl = "https://drive.google.com/thumbnail?id=1xfIjOT8-Qq4X9Sy-KyZ-mTn6YgfaFFRj";
const targetUrl = "https://drresni.com";

export const Footer: React.FC = () => {
    return (
        <footer className="w-full max-w-4xl mx-auto mt-12 mb-6">
            <a 
                href={targetUrl} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="block group rounded-lg shadow-md overflow-hidden"
            >
                <img 
                    loading="lazy"
                    src={bannerImageUrl} 
                    alt="إعلان بانر يرتبط بموقع drresni.com" 
                    className="w-full transition-transform duration-300 ease-in-out group-hover:scale-105" 
                />
            </a>
        </footer>
    );
};