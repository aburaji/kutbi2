import React from 'react';
import { Note } from '../types.ts';

interface NotesListModalProps {
    notes: Note[];
    onClose: () => void;
    onDelete: (noteId: string) => void;
}

const NotesListModal: React.FC<NotesListModalProps> = ({ notes, onClose, onDelete }) => {
    
    const sortedNotes = [...notes].sort((a, b) => b.createdAt - a.createdAt);

    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-2xl m-4 flex flex-col"
                style={{ maxHeight: '80vh' }}
                onClick={e => e.stopPropagation()}
            >
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-2xl font-bold text-slate-800">مذكراتي</h2>
                    <button onClick={onClose} className="text-slate-500 hover:text-slate-800 text-2xl font-bold">&times;</button>
                </div>
                
                <div className="flex-grow overflow-y-auto pr-2">
                    {sortedNotes.length === 0 ? (
                        <p className="text-slate-500 text-center py-10">لا يوجد مذكرات محفوظة بعد.</p>
                    ) : (
                        <div className="space-y-4">
                            {sortedNotes.map(note => (
                                <div key={note.id} className="bg-slate-50 p-4 rounded-lg border border-slate-200 flex justify-between items-start gap-4">
                                    <p className="text-slate-700 whitespace-pre-wrap flex-grow">{note.content}</p>
                                    <button 
                                        onClick={() => onDelete(note.id)} 
                                        className="text-red-500 hover:text-red-700 p-1 flex-shrink-0"
                                        aria-label="حذف المذكرة"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                          <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm4 0a1 1 0 012 0v6a1 1 0 11-2 0V8z" clipRule="evenodd" />
                                        </svg>
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
                 <div className="flex justify-end gap-4 pt-6 border-t border-slate-200 mt-4">
                    <button
                        type="button"
                        onClick={onClose}
                        className="px-6 py-2 bg-slate-200 text-slate-800 font-semibold rounded-lg hover:bg-slate-300"
                    >
                        إغلاق
                    </button>
                </div>
            </div>
        </div>
    );
};

export default NotesListModal;