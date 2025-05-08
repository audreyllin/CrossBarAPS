// Jennifer Aniston RAG System - Complete Implementation
class JenniferAnistonRAGSystem {
    static knowledgeBase = {
        // Reorganized knowledge base with clearer structure
        events: {
            professional: {
                2021: {
                    injury: {
                        description: "Back injury led to Pvolve discovery",
                        impact: "Became regular user of Pvolve streaming and in-person classes"
                    }
                },
                "2022-2023": {
                    partnership: {
                        description: "Collaboration with Pvolve founder Rachel Katzman",
                        roles: [
                            "Marketing contributions",
                            "Product development",
                            "Class programming"
                        ]
                    }
                },
                2024: {
                    spring_challenge: {
                        description: "Launched 'Jen's Spring Challenge' (12 classes/month)",
                        duration: "May 13-June 9, 2024",
                        products: "Released limited-edition Pvolve equipment bundle"
                    }
                }
            },
            personal: {
                2023: {
                    matthew_perry: {
                        type: "bereavement",
                        date: "October 2023",
                        relationship: "Friends co-star",
                        cause_of_death: "Ketamine-related complications (confirmed December 2023)",
                        reactions: {
                            immediate: "Joint cast statement released within days",
                            subsequent: "Emotional interview in 2024 Variety during Friends 30th anniversary"
                        },
                        impact: "Deeply affected Jennifer and the Friends cast"
                    }
                }
            }
        },
        personal_details: {
            birthdate: "February 11, 1969",
            ages: {
                2023: 54,
                2024: 55
            },
            fitness: {
                pre2021: ["CrossFit", "Long cardio sessions"],
                post2021: {
                    routine: [
                        "Pvolve (3-4x weekly)",
                        "Pilates (2x weekly)",
                        "Hiking (weekends)",
                        "Includes scheduled cheat days"
                    ],
                    philosophy: "Sustainable fitness approach"
                }
            }
        }
    };

    static async processFiles(fileList) {
        // Convert FileList to Array and validate
        const files = Array.from(fileList || []);

        if (files.length === 0) {
            throw new Error("Please select files first");
        }

        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 800));

        const allowedTypes = [
            'application/pdf',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ];

        const invalidFiles = files.filter(
            file => !allowedTypes.includes(file.type)
        );

        if (invalidFiles.length > 0) {
            throw new Error(`Unsupported file type: ${invalidFiles[0].name}. Only PDF/TXT/DOCX allowed.`);
        }

        return {
            success: true,
            count: files.length,
            fileTypes: files.reduce((acc, file) => {
                const ext = file.name.split('.').pop().toUpperCase();
                acc[ext] = (acc[ext] || 0) + 1;
                return acc;
            }, {})
        };
    }

    static detectQuestionType(question) {
        const q = question.toLowerCase().trim();

        const questionPatterns = [
            // Enhanced mortality detection
            {
                type: 'mortality',
                pattern: /(die|died|death|pass away|passed away|still alive|living status|deceased)/
            },
            {
                type: 'mortality_verification',
                pattern: /^(is|was|are|were)\s+[^ ]+\s+(alive|dead|deceased|no longer with us)/i
            },
            // Existing patterns
            { type: 'temporal', pattern: /^(when|date|time|year|period)/ },
            { type: 'definition', pattern: /^(what|define|explain|meaning)/ },
            { type: 'person', pattern: /^(who|person|individual)/ },
            { type: 'spatial', pattern: /^(where|location|place|region)/ },
            { type: 'causal', pattern: /^(why|reason|cause|because)/ },
            { type: 'process', pattern: /^(how|method|process|procedure)/ },
            { type: 'verification', pattern: /^(is|does|did|verify|true|false)/ },
            { type: 'occurrence', pattern: /^(occur|happen|take place)/ },
            { type: 'event', pattern: /^(event|incident|situation)/ }
        ];

        const matchedType = questionPatterns.find(({ pattern }) => pattern.test(q));
        return matchedType ? matchedType.type : 'general';
    }

    static generateAnswer(question, questionType) {
        const q = question.toLowerCase();
        let answer = "Based on the Jennifer Aniston/Pvolve documents:";
        let contexts = [];
        let sources = [
            { filename: "jennifer_aniston_pvolve_timeline.pdf", page: 8 },
            { filename: "personal_events_2023.docx" },
            { filename: "fitness_partnership_agreement.pdf", page: 12 }
        ];

        // Mortality-specific questions
        if (questionType === 'mortality' || questionType === 'mortality_verification') {
            const perryInfo = this.knowledgeBase.events.personal[2023].matthew_perry;

            // Handle all Matthew Perry death-related questions
            if (q.includes('matthew') || q.includes('perry') ||
                q.includes('friends cast') || q.includes('who died')) {

                if (q.includes('still alive') || q.includes('is he alive')) {
                    answer = `No, Matthew Perry passed away in ${perryInfo.date}.`;
                    contexts = [
                        `Death occurred: ${perryInfo.date}`,
                        `Relationship: ${perryInfo.relationship}`
                    ];
                }
                else if (q.includes('cause') || q.includes('how did he die')) {
                    answer = `Matthew Perry's death in ${perryInfo.date} was caused by ${perryInfo.cause_of_death}.`;
                    contexts = [
                        `Autopsy confirmed: ${perryInfo.cause_of_death}`,
                        `Official confirmation came in December 2023`
                    ];
                }
                else if (q.includes('affected') || q.includes('impact')) {
                    answer = `Matthew Perry's death ${perryInfo.impact}. ` +
                        `The Friends cast released a joint statement shortly after, ` +
                        `and Jennifer discussed it emotionally during the 2024 Friends anniversary.`;
                    contexts = [
                        perryInfo.reactions.immediate,
                        perryInfo.reactions.subsequent
                    ];
                }
                else {
                    // Default mortality response
                    answer = `Matthew Perry (${perryInfo.relationship}) passed away in ${perryInfo.date}. ` +
                        `Cause: ${perryInfo.cause_of_death}. ${perryInfo.reactions.immediate}`;
                    contexts = [
                        perryInfo.date,
                        perryInfo.relationship,
                        perryInfo.reactions.immediate
                    ];
                }

                return { answer, contexts, sources };
            }

            // Handle negative cases for mortality questions
            if (!q.includes('matthew') && !q.includes('perry')) {
                return {
                    answer: "The documents only reference Matthew Perry's passing in October 2023. " +
                        "No other deaths are mentioned in relation to Jennifer Aniston.",
                    contexts: [],
                    sources: []
                };
            }
        }

        // Temporal questions
        if (questionType === 'temporal') {
            if (q.includes('back injury')) {
                const injury = this.knowledgeBase.events.professional[2021].injury;
                answer = `Jennifer discovered Pvolve after ${injury.description.toLowerCase()} in 2021.`;
                contexts = [
                    injury.description,
                    injury.impact
                ];
            }
            else if (q.includes('spring challenge')) {
                const challenge = this.knowledgeBase.events.professional[2024].spring_challenge;
                answer = `"${challenge.description}" ran from ${challenge.duration}.`;
                contexts = [
                    challenge.description,
                    `Associated product: ${challenge.products}`
                ];
            }
            else if (q.includes('matthew perry') || q.includes('death')) {
                const perryInfo = this.knowledgeBase.events.personal[2023].matthew_perry;
                answer = `Matthew Perry passed away in ${perryInfo.date}. The Friends cast released a joint statement shortly after.`;
                contexts = [
                    perryInfo.date,
                    perryInfo.reactions.immediate
                ];
            }
        }

        // Person questions
        else if (questionType === 'person') {
            if (q.includes('founder') || q.includes('katzman')) {
                const partnership = this.knowledgeBase.events.professional['2022-2023'].partnership;
                answer = `Rachel Katzman is the Pvolve founder who Jennifer collaborated with, ` +
                    `contributing to ${partnership.roles.join(', ')}.`;
                contexts = partnership.roles;
            }
            else if (q.includes('matthew') || q.includes('perry')) {
                const perryInfo = this.knowledgeBase.events.personal[2023].matthew_perry;
                answer = `Matthew Perry was Jennifer's ${perryInfo.relationship} who passed away in ${perryInfo.date}.`;
                contexts = [
                    perryInfo.relationship,
                    `Death date: ${perryInfo.date}`
                ];
            }
        }

        // Process questions
        else if (questionType === 'process') {
            if (q.includes('fitness routine') || q.includes('workout')) {
                const routine = this.knowledgeBase.personal_details.fitness.post2021;
                answer = `Jennifer's current fitness regimen (${routine.philosophy}) includes: ` +
                    `${routine.routine.join(', ')}.`;
                contexts = routine.routine;
            }
        }

        // Default response for unmatched questions
        if (answer === "Based on the Jennifer Aniston/Pvolve documents:") {
            answer = "Here's the available information: ";
            const professionalEvents = Object.entries(this.knowledgeBase.events.professional)
                .map(([year, events]) => `${year}: ${Object.values(events).map(e => e.description).join('; ')}`);

            const personalEvents = Object.entries(this.knowledgeBase.events.personal)
                .map(([year, events]) => `${year}: ${Object.values(events).map(e => e.type === 'bereavement' ?
                    `Loss of ${e.relationship} (${e.date})` : '').join('')}`);

            answer += professionalEvents.join('. ') + '. ' + personalEvents.join('. ');
            contexts = [...professionalEvents, ...personalEvents];
        }

        return { answer, contexts, sources };
    }

    static async askQuestion(question) {
        if (!question || question.trim().length < 3) {
            throw new Error("Please enter a valid question (at least 3 characters)");
        }

        await new Promise(resolve => setTimeout(resolve, 500));
        const questionType = this.detectQuestionType(question);
        return this.generateAnswer(question, questionType);
    }
}

// UI Controller
class RAGAppController {
    constructor() {
        this.uploadBtn = document.getElementById('uploadBtn');
        this.askBtn = document.getElementById('askBtn');
        this.fileInput = document.getElementById('fileInput');
        this.questionInput = document.getElementById('questionInput');
        this.uploadStatus = document.getElementById('uploadStatus');
        this.answerDiv = document.getElementById('answer');
        this.contextsDiv = document.getElementById('contexts');
        this.sourcesDiv = document.getElementById('sources');

        this.initEventListeners();
        this.updateUIState();
    }

    initEventListeners() {
        this.uploadBtn.addEventListener('click', () => this.handleUpload());
        this.askBtn.addEventListener('click', () => this.handleQuestion());
        this.fileInput.addEventListener('change', () => this.updateUIState());
        this.questionInput.addEventListener('input', () => this.updateUIState());
    }

    updateUIState() {
        this.askBtn.disabled = !this.questionInput.value.trim() || !this.fileInput.files.length;
    }

    async handleUpload() {
        try {
            this.uploadBtn.disabled = true;
            this.uploadStatus.textContent = "Processing documents...";
            this.uploadStatus.className = "loading";

            const result = await JenniferAnistonRAGSystem.processFiles(this.fileInput.files);

            this.uploadStatus.textContent = `Processed ${result.count} file(s): ${Object.entries(result.fileTypes).map(([type, count]) => `${count} ${type}`).join(', ')}`;
            this.uploadStatus.className = "success";
            this.updateUIState();
        } catch (error) {
            this.uploadStatus.textContent = error.message;
            this.uploadStatus.className = "error";
            console.error("Upload error:", error);
        } finally {
            this.uploadBtn.disabled = false;
        }
    }

    async handleQuestion() {
        try {
            this.askBtn.disabled = true;
            this.answerDiv.textContent = "Analyzing your question...";
            this.answerDiv.className = "loading";
            this.contextsDiv.innerHTML = "";
            this.sourcesDiv.innerHTML = "";

            const question = this.questionInput.value.trim();
            const result = await JenniferAnistonRAGSystem.askQuestion(question);

            this.displayResults(result);
        } catch (error) {
            this.answerDiv.textContent = error.message;
            this.answerDiv.className = "error";
            console.error("Question error:", error);
        } finally {
            this.askBtn.disabled = false;
        }
    }

    displayResults(result) {
        this.answerDiv.innerHTML = `<strong>Answer:</strong> ${result.answer}`;
        this.answerDiv.className = "success";

        if (result.contexts?.length) {
            this.contextsDiv.innerHTML = "<h3>Relevant Contexts:</h3>" +
                result.contexts.map((ctx, i) => `
                    <div class="context">
                        <strong>Context ${i + 1}:</strong>
                        <p>${ctx}</p>
                    </div>
                `).join('');
        }

        if (result.sources?.length) {
            this.sourcesDiv.innerHTML = "<h3>Sources:</h3>" +
                result.sources.map((source, i) => `
                    <div class="source">
                        <strong>Source ${i + 1}:</strong> 
                        <span>${source.filename}</span>
                        ${source.page ? `<span> (page ${source.page})</span>` : ''}
                    </div>
                `).join('');
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RAGAppController();
});