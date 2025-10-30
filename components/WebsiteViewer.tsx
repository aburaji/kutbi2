import React from 'react';

const WebsiteViewer: React.FC = () => {
    return (
        <div className="p-6 bg-white rounded-2xl shadow-lg border border-slate-200">
            <div className="flex justify-between items-center mb-4 border-b pb-3 border-slate-200 flex-wrap gap-2">
                <h2 className="text-2xl font-bold text-slate-800">موقع أكاديمية درسني</h2>
            </div>
            <div className="w-full aspect-video rounded-lg overflow-hidden border border-slate-300">
                <iframe
                    src="https://drresni.com/1/"
                    title="موقع أكاديمية درسني"
                    className="w-full h-full border-0"
                    allowFullScreen
                ></iframe>
            </div>
        </div>
    );
};

export default WebsiteViewer;
