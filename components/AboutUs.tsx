import React, { useState } from 'react';

const AccordionItem: React.FC<{
    title: string;
    isOpen: boolean;
    onClick: () => void;
    content: React.ReactNode;
}> = ({ title, isOpen, onClick, content }) => {
    return (
        <div className="border-b border-slate-200">
            <h2>
                <button
                    type="button"
                    className="flex justify-between items-center w-full py-5 font-semibold text-right text-slate-800"
                    onClick={onClick}
                    aria-expanded={isOpen}
                >
                    <span>{title}</span>
                    <svg
                        className={`w-6 h-6 transform transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                    </svg>
                </button>
            </h2>
            <div
                className={`grid transition-all duration-300 ease-in-out ${isOpen ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}
            >
                <div className="overflow-hidden">
                    <div className="py-5 px-2 text-slate-600 space-y-4">
                        {content}
                    </div>
                </div>
            </div>
        </div>
    );
};

const AboutUs: React.FC = () => {
    const [openItem, setOpenItem] = useState<string | null>('definition'); // Open first item by default

    const accordionItems = [
        { 
            id: 'definition', 
            title: 'تعريف بالمبادرة',
            content: (
                <>
                    <p>
                        <strong>"الكُتُبي الذكي"</strong> هو أحد المشاريع الرائدة ضمن <strong>"مبادرة أكاديمية درسني"</strong>، وهي المبادرة التعليمية العالمية لمركز الوسطية للبحوث والدراسات.
                    </p>
                    <p>
                        يهدف هذا المشروع إلى توفير كُتُبي ذكي لطلاب العلم والباحثين، يتيح لهم تحليل الكتب والمواد العلمية ودراستها بعمق باستخدام أحدث تقنيات الذكاء الاصطناعي من Gemini. نحن نسعى لتمكين الجيل القادم من العلماء والمفكرين بأدوات مبتكرة تسهل عليهم رحلة طلب العلم وتحصيله.
                    </p>
                </>
            )
        },
        { 
            id: 'projects', 
            title: 'مشاريعنا الأخرى',
            content: (
                 <>
                    <p>
                        نعمل في "أكاديمية درسني" على مجموعة متكاملة من المشاريع التي تخدم العلم وطلابه، منها:
                    </p>
                    <ul className="list-disc list-inside space-y-2 pr-4">
                        <li><strong>منصة درسني التعليمية:</strong> بيئة تعليمية تفاعلية تقدم دورات متخصصة في العلوم الشرعية والعربية.</li>
                        <li><strong>المكتبة الرقمية الشاملة:</strong> مشروع لجمع وفهرسة وتوفير أمهات الكتب والمخطوطات النادرة للباحثين حول العالم.</li>
                        <li><strong>برنامج إعداد العلماء:</strong> منهج متكامل لإعداد وتأهيل الكوادر العلمية المتمكنة في مختلف التخصصات.</li>
                        <li><strong>جراب علم:</strong> سلسلة من المحتويات التعليمية المركزة التي تهدف إلى تزويد طالب العلم بما لا يسعه جهله.</li>
                    </ul>
                </>
            )
        },
        { 
            id: 'support', 
            title: 'ادعمنا',
            content: (
                 <>
                    <p>
                        مبادراتنا قائمة على الجهود الذاتية والتبرعات. دعمكم يساهم في استمرارية هذه المشاريع وتطويرها ووصولها لأكبر عدد ممكن من المستفيدين. يمكنكم دعمنا من خلال:
                    </p>
                     <ul className="list-disc list-inside space-y-2 pr-4">
                        <li><strong>الدعاء:</strong> نسألكم الدعاء لنا بالتوفيق والسداد.</li>
                        <li><strong>النشر:</strong> مساهمتكم في نشر تطبيقاتنا ومشاريعنا بين المهتمين.</li>
                        <li><strong>الملاحظات:</strong> تزويدنا بآرائكم ومقترحاتكم لتطوير خدماتنا.</li>
                        <li><strong>الدعم المادي:</strong> لدعم مشاريعنا مالياً، يرجى التواصل معنا عبر البريد الإلكتروني: <a href="mailto:support@example.com" className="text-blue-600 hover:underline">support@example.com</a></li>
                    </ul>
                </>
            )
        },
    ];

    const handleToggle = (id: string) => {
        setOpenItem(openItem === id ? null : id);
    };

    return (
        <div className="p-6 bg-white rounded-2xl shadow-lg border border-slate-200">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-slate-800">من نحن</h2>
                <p className="mt-2 text-slate-600">تعرف على المزيد عنا وعن مبادراتنا ورؤيتنا.</p>
            </div>
            
            <div className="max-w-2xl mx-auto">
                {accordionItems.map(item => (
                    <AccordionItem
                        key={item.id}
                        title={item.title}
                        isOpen={openItem === item.id}
                        onClick={() => handleToggle(item.id)}
                        content={item.content}
                    />
                ))}
            </div>
        </div>
    );
};

export default AboutUs;
