import React, { useState } from 'react';
import { Audio } from '../types.ts';
import StarRating from './StarRating.tsx';

const MediaCover: React.FC<{ item: Audio; onZoomRequest: (imageUrl: string) => void; }> = ({ item, onZoomRequest }) => {
    return (
        <button
            onClick={() => onZoomRequest(item.thumbnailUrl)}
            className="w-full h-80 block overflow-hidden bg-slate-200 flex items-center justify-center"
            aria-label={`عرض غلاف ${item.title}`}
        >
            <img 
                loading="lazy"
                src={item.thumbnailUrl} 
                alt={`غلاف ${item.title}`}
                className="h-full w-full object-cover group-hover:scale-105 transition-transform duration-300 ease-in-out"
                onError={(e) => e.currentTarget.src = 'https://placehold.co/400x600/1e293b/ffffff?text=Audio'}
            />
        </button>
    );
};

interface SelectorProps {
    items: Audio[];
    onAnalyze: (item: Audio) => void;
    isAnalyzing: boolean;
    analyzingItemId: string | null;
    onOpenAddModal: () => void;
    onOpenNotesListModal: () => void;
    editingId: string | null;
    onStartEdit: (id: string | null) => void;
    onSaveEdit: (id: string, newTitle: string) => void;
    editingDescId: string | null;
    onStartDescEdit: (id: string | null) => void;
    onSaveDescEdit: (id: string, newDescription: string) => void;
    onDelete: (item: Audio) => void;
    onRate: (id: string, newRating: number) => void;
}

const AudioSelector: React.FC<SelectorProps> = ({ 
    items, onAnalyze, isAnalyzing, analyzingItemId, onOpenAddModal, onOpenNotesListModal,
    editingId, onStartEdit, onSaveEdit, editingDescId, onStartDescEdit, onSaveDescEdit, onDelete, onRate
}) => {
    const [zoomedImage, setZoomedImage] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');

    const handleTitleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, id: string) => {
        // FIX: Use e.currentTarget to ensure correct typing.
        if (e.key === 'Enter') onSaveEdit(id, e.currentTarget.value);
        else if (e.key === 'Escape') onStartEdit(null);
    };

    const handleDescriptionKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>, id: string) => {
        // FIX: Use e.currentTarget to ensure correct typing.
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSaveDescEdit(id, e.currentTarget.value); } 
        else if (e.key === 'Escape') onStartDescEdit(null);
    };

    const filteredItems = items.filter(item =>
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (item.description && item.description.toLowerCase().includes(searchQuery.toLowerCase()))
    );

    return (
        <div className="p-6 bg-white rounded-2xl shadow-lg border border-slate-200">
            <div className="flex justify-between items-center mb-4 border-b pb-3 border-slate-200 flex-wrap gap-2">
                <h2 className="text-2xl font-bold text-slate-800">مكتبة الصوتيات</h2>
                <div className="flex items-center gap-3">
                    <button onClick={onOpenNotesListModal} className="flex items-center gap-2 bg-white text-slate-700 font-semibold py-2 px-4 rounded-lg hover:bg-slate-50 transition-colors duration-200 text-sm shadow-sm border border-slate-300">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                        <span>مذكراتي</span>
                    </button>
                    <button onClick={onOpenAddModal} className="flex items-center gap-2 bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors duration-200 text-sm shadow hover:shadow-md">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110 2h3V6a1 1 0 011-1z" clipRule="evenodd" /></svg>
                        <span>إضافة ملف صوتي</span>
                    </button>
                </div>
            </div>
            
            <div className="my-4 relative">
                <input type="search" placeholder="ابحث..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="w-full pl-10 pr-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-400" />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-slate-400" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" /></svg></div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8 pt-4">
                {filteredItems.length > 0 ? filteredItems.map(item => {
                    if (item.isLoading) {
                        return (
                             <div key={item.id} className="group flex flex-col bg-slate-100 rounded-xl shadow-md overflow-hidden border border-slate-200">
                                <div className="w-full h-80 block overflow-hidden bg-slate-200 animate-pulse relative flex flex-col items-center justify-center">
                                    <svg className="animate-spin h-10 w-10 text-slate-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                    <p className="text-slate-500 font-semibold mt-4">جاري إضافة الصوت...</p>
                                </div>
                                <div className="p-5 flex flex-col flex-grow"><div className="h-6 bg-slate-200 rounded w-3/4 mb-2 animate-pulse"></div><div className="flex-grow min-h-[60px] space-y-2"><div className="h-4 bg-slate-200 rounded w-full animate-pulse"></div><div className="h-4 bg-slate-200 rounded w-5/6 animate-pulse"></div></div><div className="mt-auto pt-4 border-t border-slate-200 space-y-3"><div className="flex items-center gap-2"><div className="flex-1 h-9 bg-slate-200 rounded-lg animate-pulse"></div><div className="flex-1 h-9 bg-slate-200 rounded-lg animate-pulse"></div></div><div className="w-full h-10 bg-slate-300 rounded-lg animate-pulse"></div></div></div>
                            </div>
                        );
                    }
                    return (
                        <div key={item.id} className="group flex flex-col bg-slate-50 rounded-xl shadow-md hover:shadow-2xl transition-all duration-300 border border-slate-200 hover:border-blue-400 overflow-hidden">
                            <div className="relative"><MediaCover item={item} onZoomRequest={setZoomedImage} /><div className="absolute top-3 right-3 flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity bg-black bg-opacity-40 p-1 rounded-lg"><button onClick={() => editingId === item.id ? onStartEdit(null) : onStartEdit(item.id)} data-tooltip="تعديل العنوان" className="text-white hover:text-blue-300 p-1.5"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z" /><path fillRule="evenodd" d="M2 6a2 2 0 012-2h4a1 1 0 010 2H4v10h10v-4a1 1 0 112 0v4a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" clipRule="evenodd" /></svg></button><button onClick={() => onDelete(item)} data-tooltip="حذف" className="text-white hover:text-red-400 p-1.5"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z" clipRule="evenodd" /></svg></button></div></div>
                            <div className="p-5 flex flex-col flex-grow">
                                {editingId === item.id ? <input type="text" defaultValue={item.title} onKeyDown={(e) => handleTitleKeyDown(e, item.id)} onBlur={(e) => onSaveEdit(item.id, e.currentTarget.value)} className="font-bold text-lg text-slate-800 bg-white border border-blue-400 rounded px-2 py-1 w-full mb-2" autoFocus /> : <h3 className="font-bold text-lg text-slate-800 group-hover:text-blue-600 truncate mb-2" title={item.title}>{item.title}</h3>}
                                <div className="flex-grow min-h-[60px]">{editingDescId === item.id ? <textarea defaultValue={item.description} onKeyDown={(e) => handleDescriptionKeyDown(e, item.id)} onBlur={(e) => onSaveDescEdit(item.id, e.currentTarget.value)} className="text-sm text-slate-600 bg-white border border-blue-400 rounded px-2 py-1 w-full" autoFocus rows={3} /> : <div className="flex items-start gap-1"><p className="text-sm text-slate-600 line-clamp-2 flex-grow">{item.description}</p><button onClick={() => editingDescId === item.id ? onStartDescEdit(null) : onStartDescEdit(item.id)} data-tooltip="تعديل الوصف" className="text-slate-400 hover:text-blue-600 p-1 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"><svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z" /><path fillRule="evenodd" d="M2 6a2 2 0 012-2h4a1 1 0 010 2H4v10h10v-4a1 1 0 112 0v4a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" clipRule="evenodd" /></svg></button></div>}</div>
                                <div className="flex items-center justify-between my-3"><StarRating rating={item.rating || 0} onRate={(r) => onRate(item.id, r)} />{item.rating ? <span className="text-xs font-bold text-slate-500 bg-slate-200 px-2 py-0.5 rounded-full">تقييمك: {item.rating.toFixed(1)}</span> : <span className="text-xs text-slate-400">قيّم بنفسك</span>}</div>
                                <div className="mt-auto pt-4 border-t border-slate-200 space-y-3">
                                    <div className="flex items-center gap-2">
                                        <a href={item.downloadUrl || '#'} target="_blank" rel="noopener noreferrer" onClick={(e) => !item.downloadUrl && e.preventDefault()} className={`flex-1 flex items-center justify-center text-center border border-sky-600 text-sky-600 font-semibold py-2 px-3 rounded-lg hover:bg-sky-50 transition-colors duration-300 text-sm ${!item.downloadUrl ? 'opacity-50 cursor-not-allowed' : ''}`}>
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" /></svg>
                                            <span className="mr-2">استماع</span>
                                        </a>
                                        <a href={item.downloadUrl || '#'} download={item.fileName} target="_blank" rel="noopener noreferrer" onClick={(e) => !item.downloadUrl && e.preventDefault()} className={`flex-1 flex items-center justify-center text-center border border-green-600 text-green-600 font-semibold py-2 px-3 rounded-lg hover:bg-green-50 transition-colors duration-300 text-sm ${!item.downloadUrl ? 'opacity-50 cursor-not-allowed' : ''}`}>
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" /></svg>
                                            <span className="mr-2">تحميل</span>
                                        </a>
                                    </div>
                                    <button onClick={() => onAnalyze(item)} disabled={true} className="w-full bg-slate-400 text-white font-semibold py-2.5 px-4 rounded-lg cursor-not-allowed flex justify-center items-center" data-tooltip="التحليل متاح للكتب والفيديوهات فقط حاليًا">
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" /></svg>
                                        <span className="mr-2">تحليل</span>
                                    </button>
                                </div>
                            </div>
                        </div>
                    );
                }) : <div className="col-span-full text-center py-12 text-slate-500"><h3 className="text-xl font-semibold">لا توجد ملفات صوتية</h3><p className="mt-2">انقر على "إضافة ملف صوتي" لتبدأ.</p></div>}
            </div>

            {zoomedImage && (<div className="fixed inset-0 bg-black bg-opacity-80 flex justify-center items-center z-50 p-4" onClick={() => setZoomedImage(null)}><div className="relative max-w-[90vw] max-h-[90vh] flex" onClick={(e) => e.stopPropagation()}><img src={zoomedImage} alt="صورة مكبرة" className="object-contain rounded-lg shadow-2xl" /><button onClick={() => setZoomedImage(null)} className="absolute -top-3 -right-3 bg-white text-slate-800 rounded-full h-9 w-9 flex items-center justify-center text-2xl font-bold" aria-label="إغلاق">&times;</button></div></div>)}
        </div>
    );
};

export default AudioSelector;