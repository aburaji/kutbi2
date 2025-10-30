import React from 'react';
import { Video } from '../types.ts';

interface ScriptViewerModalProps {
    video: Video;
    onClose: () => void;
}

const ScriptViewerModal: React.FC<ScriptViewerModalProps> = ({ video, onClose }) => {
    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-60 flex justify-center items-center z-50 p-4"
            onClick={onClose}
        >
            <div 
                className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-2xl flex flex-col"
                style={{ maxHeight: '85vh' }}
                onClick={e => e.stopPropagation()}
            >
                <div className="flex justify-between items-center mb-6 pb-4 border-b border-slate-200">
                    <h2 className="text-2xl font-bold text-slate-800">نص الفيديو: {video.title}</h2>
                    <button onClick={onClose} className="text-slate-500 hover:text-slate-800 text-3xl font-bold leading-none">&times;</button>
                </div>
                
                <div className="flex-grow overflow-y-auto pr-2">
                    <p className="text-slate-700 whitespace-pre-wrap leading-relaxed">
                        {video.script}
                    </p>
                </div>

                 <div className="flex justify-end pt-6 mt-4 border-t border-slate-200">
                    <button
                        type="button"
                        onClick={onClose}
                        className="px-6 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700"
                    >
                        إغلاق
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ScriptViewerModal;