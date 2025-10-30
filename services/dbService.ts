import { Book, Note, Video, Audio, Image } from '../types.ts';

const DB_NAME = 'BookLibraryDB';
const DB_VERSION = 4; // Incremented version
const BOOK_STORE_NAME = 'userBooks';
const RESEARCH_STORE_NAME = 'userResearches';
const PERIODICAL_STORE_NAME = 'userPeriodicals';
const NOTE_STORE_NAME = 'userNotes';
// FIX: Added missing store names for video, audio, and image types.
const VIDEO_STORE_NAME = 'userVideos';
const AUDIO_STORE_NAME = 'userAudios';
const IMAGE_STORE_NAME = 'userImages';


class DbService {
    private db: IDBDatabase | null = null;
    private initPromise: Promise<void> | null = null;

    constructor() {
        this.initPromise = this.init();
    }

    private async init(): Promise<void> {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);

            request.onupgradeneeded = (event) => {
                const db = (event.target as IDBOpenDBRequest).result;
                if (!db.objectStoreNames.contains(BOOK_STORE_NAME)) {
                    db.createObjectStore(BOOK_STORE_NAME, { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains(RESEARCH_STORE_NAME)) {
                    db.createObjectStore(RESEARCH_STORE_NAME, { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains(PERIODICAL_STORE_NAME)) {
                    db.createObjectStore(PERIODICAL_STORE_NAME, { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains(NOTE_STORE_NAME)) {
                    db.createObjectStore(NOTE_STORE_NAME, { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains(VIDEO_STORE_NAME)) {
                    db.createObjectStore(VIDEO_STORE_NAME, { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains(AUDIO_STORE_NAME)) {
                    db.createObjectStore(AUDIO_STORE_NAME, { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains(IMAGE_STORE_NAME)) {
                    db.createObjectStore(IMAGE_STORE_NAME, { keyPath: 'id' });
                }
            };

            request.onsuccess = (event) => {
                this.db = (event.target as IDBOpenDBRequest).result;
                resolve();
            };

            request.onerror = (event) => {
                console.error('IndexedDB error:', (event.target as IDBOpenDBRequest).error);
                reject('Failed to open IndexedDB.');
            };
        });
    }
    
    private async getStore(storeName: string, mode: IDBTransactionMode): Promise<IDBObjectStore> {
        if (!this.initPromise) {
            this.initPromise = this.init();
        }
        await this.initPromise;
        if (!this.db) {
            throw new Error("Database not initialized.");
        }
        const transaction = this.db.transaction(storeName, mode);
        return transaction.objectStore(storeName);
    }

    private async getAll<T>(storeName: string): Promise<T[]> {
        const store = await this.getStore(storeName, 'readonly');
        return new Promise((resolve, reject) => {
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    private async add<T>(storeName: string, item: T): Promise<void> {
        const store = await this.getStore(storeName, 'readwrite');
        return new Promise((resolve, reject) => {
            const request = store.add(item);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    private async update<T>(storeName: string, item: T): Promise<void> {
        const store = await this.getStore(storeName, 'readwrite');
        return new Promise((resolve, reject) => {
            const request = store.put(item);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    private async delete(storeName: string, id: string): Promise<void> {
        const store = await this.getStore(storeName, 'readwrite');
        return new Promise((resolve, reject) => {
            const request = store.delete(id);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    // Books
    getAllUserBooks = () => this.getAll<Book>(BOOK_STORE_NAME);
    addBook = (book: Book) => this.add(BOOK_STORE_NAME, book);
    updateBook = (book: Book) => this.update(BOOK_STORE_NAME, book);
    deleteBook = (id: string) => this.delete(BOOK_STORE_NAME, id);

    // Researches
    getAllUserResearches = () => this.getAll<Book>(RESEARCH_STORE_NAME);
    addResearch = (research: Book) => this.add(RESEARCH_STORE_NAME, research);
    updateResearch = (research: Book) => this.update(RESEARCH_STORE_NAME, research);
    deleteResearch = (id: string) => this.delete(RESEARCH_STORE_NAME, id);
    
    // Periodicals
    getAllUserPeriodicals = () => this.getAll<Book>(PERIODICAL_STORE_NAME);
    addPeriodical = (periodical: Book) => this.add(PERIODICAL_STORE_NAME, periodical);
    updatePeriodical = (periodical: Book) => this.update(PERIODICAL_STORE_NAME, periodical);
    deletePeriodical = (id: string) => this.delete(PERIODICAL_STORE_NAME, id);

    // Videos
    getAllUserVideos = () => this.getAll<Video>(VIDEO_STORE_NAME);
    addVideo = (video: Video) => this.add(VIDEO_STORE_NAME, video);
    updateVideo = (video: Video) => this.update(VIDEO_STORE_NAME, video);
    deleteVideo = (id: string) => this.delete(VIDEO_STORE_NAME, id);

    // Audios
    getAllUserAudios = () => this.getAll<Audio>(AUDIO_STORE_NAME);
    addAudio = (audio: Audio) => this.add(AUDIO_STORE_NAME, audio);
    updateAudio = (audio: Audio) => this.update(AUDIO_STORE_NAME, audio);
    deleteAudio = (id: string) => this.delete(AUDIO_STORE_NAME, id);

    // Images
    getAllUserImages = () => this.getAll<Image>(IMAGE_STORE_NAME);
    addImage = (image: Image) => this.add(IMAGE_STORE_NAME, image);
    updateImage = (image: Image) => this.update(IMAGE_STORE_NAME, image);
    deleteImage = (id: string) => this.delete(IMAGE_STORE_NAME, id);
    
    // Notes
    getAllNotes = () => this.getAll<Note>(NOTE_STORE_NAME);
    addNote = (note: Note) => this.add(NOTE_STORE_NAME, note);
    deleteNote = (id: string) => this.delete(NOTE_STORE_NAME, id);
}

let dbServiceInstance: DbService | null = null;

// FIX: Added the exported 'getDbService' function to provide a singleton instance of the DbService.
export const getDbService = (): DbService => {
    if (!dbServiceInstance) {
        dbServiceInstance = new DbService();
    }
    return dbServiceInstance;
};