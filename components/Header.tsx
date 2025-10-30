import React from 'react';

const Header: React.FC = () => {
    return (
        <header className="w-full text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-extrabold text-slate-800 mb-1">
                الكُتُبي الذكي
            </h1>
            <p className="text-3xl md:text-4xl font-semibold text-slate-700 mb-6">
                جراب علم
            </p>
            <p className="text-lg text-slate-600">
                برنامج إعداد العلماء ومالا يسع الشيخ جهله
            </p>
            <p className="text-md text-slate-500 mt-2">
                المبادرة التعليمية العالمية لمركز الوسطية للبحوث والدراسات
            </p>
        </header>
    );
}

export default Header;