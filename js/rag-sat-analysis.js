class EnhancedRAGSystem {
    constructor() {
        this.textCache = new Map();
        this.questionHandlers = {
            main_idea: this.handleMainIdeaIQuestion,
            logical_completion: this.handleLogicalCompletion,
            evidence_support: this.handleEvidenceSupport,
            detail_extraction: this.handleDetailExtraction,
            comparative: this.handleComparativeQuestion,
            general_comprehension: this.handleGeneralQuestion
        };
    }

    // Core processing function
    async processQuery(question, contextText) {
        // Step 1: Analyze question type
        const questionType = this.detectQuestionType(question);

        // Step 2: Analyze text structure
        const textStructures = this.analyzeTextStructure(contextText);

        // Step 3: Extract relevant context based on question type
        const relevantContext = this.extractRelevantContext(questionType, contextText, textStructures);

        // Step 4: Generate potential answers (in a real system, this would come from LLM)
        const potentialAnswers = this.generatePotentialAnswers(question, relevantContext);

        // Step 5: Validate and score answers
        const scoredAnswers = potentialAnswers.map(answer => ({
            text: answer,
            score: this.scoreAnswerConfidence(answer, relevantContext, questionType),
            validation: this.validateAnswer(relevantContext, questionType, answer)
        }));

        // Step 6: Select and format best answer
        return this.formatBestAnswer(scoredAnswers, questionType);
    }

    // Question Type Detection (enhanced)
    detectQuestionType(question) {
        const q = question.toLowerCase().trim();

        // Main idea questions
        if (q.includes("main idea") || q.includes("best states") ||
            q.includes("primary purpose") || q.includes("central theme"))
            return "main_idea";

        // Logical completion
        if (q.includes("most logically completes") || q.includes("should recognize") ||
            q.includes("most logically follow") || q.match(/blank$/))
            return "logical_completion";

        // Evidence support
        if (q.includes("most strongly support") || q.includes("best support") ||
            q.includes("best evidence") || q.includes("would justify"))
            return "evidence_support";

        // Detail extraction
        if (q.includes("based on the text") || q.includes("according to the text") ||
            q.includes("the text indicates") || q.includes("can be concluded"))
            return "detail_extraction";

        // Comparative analysis
        if (q.includes("more likely") || q.includes("primary difference") ||
            q.includes("compared to") || q.includes("contrast"))
            return "comparative";

        return "general_comprehension";
    }

    // Text Structure Analysis (enhanced)
    analyzeTextStructure(text) {
        const structures = [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);

        // Detect claim-evidence patterns
        const claimEvidenceMarkers = /however|although|but.+(study|research|evidence|data|findings)/i;
        if (sentences.some(s => claimEvidenceMarkers.test(s)))
            structures.push("claim_evidence");

        // Detect comparative structures
        const comparativeMarkers = /whereas|while|in contrast|on the other hand|compared to|similar to/i;
        if (sentences.some(s => comparativeMarkers.test(s)))
            structures.push("comparative");

        // Detect problem-solution
        const problemSolutionMarkers = /(problem|issue|challenge|difficulty).+(solution|resolved|address|solve|remedy)/i;
        if (sentences.some(s => problemSolutionMarkers.test(s)))
            structures.push("problem_solution");

        // Detect process description
        const processMarkers = /(first|then|next|finally|after|process|method|experiment|steps?)\b/i;
        if (sentences.some(s => processMarkers.test(s)))
            structures.push("process");

        // Detect chronological structure
        const timeMarkers = /(initially|previously|subsequently|prior to|during|afterward)/i;
        if (sentences.some(s => timeMarkers.test(s)))
            structures.push("chronological");

        return structures.length ? structures : ["general_exposition"];
    }

    // Context Extraction
    extractRelevantContext(questionType, text, structures) {
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        let relevantSentences = [];

        switch (questionType) {
            case "main_idea":
                // Prioritize opening, closing, and repeated concepts
                relevantSentences.push(sentences[0]); // First sentence often topic
                relevantSentences.push(sentences[sentences.length - 1]); // Last often conclusion

                // Add sentences with repeated nouns/proper nouns
                const nounCounts = this.countNouns(text);
                const topNouns = Object.entries(nounCounts)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 3)
                    .map(e => e[0]);

                relevantSentences.push(...sentences.filter(s =>
                    topNouns.some(noun => new RegExp(`\\b${noun}\\b`, 'i').test(s))
                );
                break;

            case "evidence_support":
                // Focus on research findings, statistics, quotes
                relevantSentences = sentences.filter(s =>
                    /(study|research|experiment|found|discovered|according to)\b/i.test(s) ||
                    /\d+%/.test(s) || /"[^"]+"/.test(s)
                );
                break;

            case "logical_completion":
                // Focus on the last few sentences and argument flow
                relevantSentences = sentences.slice(-3);

                // Add any "therefore", "thus", "hence" markers
                relevantSentences.push(...sentences.filter(s =>
                    /\b(therefore|thus|hence|so|consequently)\b/i.test(s))
                );
                break;

            case "comparative":
                // Focus on comparison markers and relative terms
                relevantSentences = sentences.filter(s =>
                    /\b(whereas|while|compared to|similar to|different from|more|less)\b/i.test(s)
                );
                break;

            default:
                // For detail extraction and general, use all sentences but weighted
                relevantSentences = sentences;
        }

        // Further refine based on text structures
        if (structures.includes("claim_evidence")) {
            relevantSentences = relevantSentences.filter(s =>
                /(however|although|but|yet)\b/i.test(s) ||
                /(evidence|data|shows|demonstrates)\b/i.test(s)
            );
        }

        if (structures.includes("problem_solution")) {
            relevantSentences = relevantSentences.filter(s =>
                /(problem|issue|challenge|solution|resolve|address)\b/i.test(s)
            );
        }

        return relevantSentences.join(". ") + ".";
    }

    // Answer Validation
    validateAnswer(context, questionType, answer) {
        const criteria = {
            main_idea: {
                must: ["cover primary focus", "not overgeneralize"],
                must_not: ["focus on minor details", "introduce new concepts"],
                score: 0
            },
            logical_completion: {
                must: ["follow argument flow", "maintain coherence"],
                must_not: ["introduce new topics", "contradict premises"],
                score: 0
            },
            evidence_support: {
                must: ["directly support claim", "be textually grounded"],
                must_not: ["weaken argument", "be irrelevant"],
                score: 0
            },
            detail_extraction: {
                must: ["be textually explicit", "match scope"],
                must_not: ["require assumptions", "be too broad/narrow"],
                score: 0
            },
            comparative: {
                must: ["identify correct relationship", "maintain comparison"],
                must_not: ["reverse relationship", "ignore key differences"],
                score: 0
            }
        };

        const rules = criteria[questionType];

        // Check positive criteria
        if (rules.must.includes("cover primary focus") &&
            this.checkPrimaryFocusCoverage(context, answer))
            rules.score += 1;

        if (rules.must.includes("be textually grounded") &&
            this.checkTextualGrounding(context, answer))
            rules.score += 1;

        // Check negative criteria
        if (rules.must_not.includes("introduce new concepts") &&
            this.checkNewConcepts(context, answer))
            rules.score -= 1;

        if (rules.must_not.includes("be irrelevant") &&
            this.checkIrrelevance(context, answer))
            rules.score -= 1;

        return rules.score;
    }

    // Confidence Scoring
    scoreAnswerConfidence(answer, context, questionType) {
        let score = 0;

        // 1. Textual overlap (30% weight)
        score += 0.3 * this.calculateTermOverlap(answer, context);

        // 2. Logical consistency (30% weight)
        score += 0.3 * this.analyzeLogicalFlow(context, answer);

        // 3. Scope appropriateness (20% weight)
        score += 0.2 * this.evaluateScopeMatch(context, answer, questionType);

        // 4. Distractor analysis (20% weight)
        score += 0.2 * (1 - this.identifyCommonDistractorPatterns(answer));

        // Normalize to 0-1 range
        return Math.min(1, Math.max(0, score));
    }

    // Helper Methods
    countNouns(text) {
        // Simplified noun counter (in real implementation would use NLP)
        const words = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
        const commonNouns = new Set(['study', 'research', 'data', 'result', 'effect',
            'author', 'theory', 'hypothesis', 'method', 'conclusion']);
        const properNouns = text.match(/\b[A-Z][a-z]+\b/g) || [];

        const allNouns = [...words.filter(w => commonNouns.has(w)), ...properNouns];
        return allNouns.reduce((counts, noun) => {
            counts[noun] = (counts[noun] || 0) + 1;
            return counts;
        }, {});
    }

    calculateTermOverlap(answer, context) {
        const answerTerms = new Set(answer.toLowerCase().split(/\s+/));
        const contextTerms = new Set(context.toLowerCase().split(/\s+/));
        const intersection = [...answerTerms].filter(term => contextTerms.has(term));
        return intersection.length / answerTerms.size;
    }

    analyzeLogicalFlow(context, answer) {
        // Simplified logic analysis (would use more sophisticated NLP in production)
        const contextSentences = context.split(/[.!?]+/).filter(s => s.trim());
        const lastContextSentence = contextSentences[contextSentences.length - 1];

        // Check for continuation markers
        const continuationMarkers = /^(and|but|however|therefore|thus|so|because)/i;
        if (continuationMarkers.test(answer.trim())) {
            const precedingWord = continuationMarkers.exec(answer.trim())[1].toLowerCase();

            if (precedingWord === 'and' && !/and\b/i.test(lastContextSentence))
                return 0.8;
            if (precedingWord === 'but' && !/but\b/i.test(lastContextSentence))
                return 0.9;
            if (/therefore|thus|so/.test(precedingWord) && !/because\b/i.test(lastContextSentence))
                return 0.7;
        }

        // Default score based on term overlap with last sentence
        return this.calculateTermOverlap(answer, lastContextSentence);
    }

    evaluateScopeMatch(context, answer, questionType) {
        const contextScope = this.determineScope(context);
        const answerScope = this.determineScope(answer);

        // For detail extraction, answer scope should be equal or narrower
        if (questionType === "detail_extraction") {
            return answerScope <= contextScope ? 1 : 0.5;
        }

        // For main idea, answer scope should match
        if (questionType === "main_idea") {
            return Math.abs(answerScope - contextScope) < 0.2 ? 1 : 0.3;
        }

        // Default case
        return 1 - Math.min(1, Math.abs(answerScope - contextScope)));
    }

    determineScope(text) {
        // Simplified scope measurement (0-1)
        const sentences = text.split(/[.!?]+/).filter(s => s.trim());
        const wordCount = sentences.reduce((sum, s) => sum + s.split(/\s+/).length, 0);
        const avgSentenceLength = wordCount / sentences.length;

        // Count "big" words (3+ syllables)
        const bigWords = text.split(/\s+/).filter(w => w.length > 8).length;
        const bigWordRatio = bigWords / wordCount;

        // Count qualifiers
        const qualifiers = text.match(/\b(some|many|often|frequently|usually|generally)\b/gi) || [];
        const qualifierRatio = qualifiers.length / wordCount;

        return Math.min(1, (avgSentenceLength / 20 + bigWordRatio * 2 + qualifierRatio * 3) / 3);
    }

    identifyCommonDistractorPatterns(answer) {
        // Patterns that often indicate incorrect answers
        const distractorPatterns = [
            /most (important|effective|significant)/i, // Absolute statements
            /all \w+ are/i, // Overgeneralizations
            /none of the/i, // Extreme negatives
            /completely \w+/i, // Absolutes
            /entirely \w+/i, // Absolutes
            /^[^.]+\?$/ // Question-form answers
        ];

        return distractorPatterns.some(pattern => pattern.test(answer)) ? 1 : 0;
    }

    // Question Handlers
    handleMainIdeaIQuestion(context) {
        const sentences = context.split(/[.!?]+/).filter(s => s.trim());
        const firstSentence = sentences[0];
        const lastSentence = sentences[sentences.length - 1];

        // Simple extraction - in real system would use more sophisticated summarization
        return `The main idea is ${firstSentence}. This is supported by ${lastSentence}.`;
    }

    handleLogicalCompletion(context) {
        const sentences = context.split(/[.!?]+/).filter(s => s.trim());
        const lastSentence = sentences[sentences.length - 1];

        // Analyze the direction of the last sentence
        if (/however|but|although/i.test(lastSentence)) {
            return "This suggests a contrasting idea would follow.";
        }
        if (/because|since|as/i.test(lastSentence)) {
            return "This indicates a reason or explanation would follow.";
        }
        if (/therefore|thus|so/i.test(lastSentence)) {
            return "This suggests a conclusion or result would follow.";
        }

        return "This suggests additional supporting information would follow.";
    }

    // Formatting the final answer
    formatBestAnswer(scoredAnswers, questionType) {
        // Sort by score descending
        scoredAnswers.sort((a, b) => b.score - a.score);
        const bestAnswer = scoredAnswers[0];

        return {
            answer: bestAnswer.text,
            confidence: bestAnswer.score.toFixed(2),
            explanation: this.generateExplanation(bestAnswer, questionType),
            alternatives: scoredAnswers.slice(1, 3).map(a => ({
                text: a.text,
                score: a.score.toFixed(2)
            }))
        };
    }

    generateExplanation(answer, questionType) {
        const explanations = {
            main_idea: `This answer best captures the primary focus of the text with a confidence of ${answer.score.toFixed(2)}.`,
            logical_completion: `This option most coherently continues the text's logical flow (confidence: ${answer.score.toFixed(2)}).`,
            evidence_support: `This statement provides the strongest direct support for the claim (confidence: ${answer.score.toFixed(2)}).`,
            detail_extraction: `This conclusion is most directly supported by textual evidence (confidence: ${answer.score.toFixed(2)}).`,
            comparative: `This option correctly identifies the key relationship (confidence: ${answer.score.toFixed(2)}).`
        };

        return explanations[questionType] || `Selected with confidence ${answer.score.toFixed(2)} based on textual analysis.`;
    }
}

// Usage Example:
const ragSystem = new EnhancedRAGSystem();
const context = "Paleontologists searching for signs of ancient life have found many fossilized specimens...";
const question = "Based on the text, what can be concluded about the geological formation at Mistaken Point?";

ragSystem.processQuery(question, context)
    .then(result => {
        console.log("Answer:", result.answer);
        console.log("Confidence:", result.confidence);
        console.log("Explanation:", result.explanation);
    });