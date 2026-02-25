# BDH Tutorial Platform

An interactive tutorial for the **Baby Dragon Hatchling (BDH)** architecture — a biologically plausible post-Transformer AI system.

Built with React + Vite, inspired by LangChain Docs UI + Kaggle/Coursera interactive exercises.

## Tech Stack

| Layer | Choice |
|-------|--------|
| Framework | React 18 + Vite 5 |
| Routing | React Router v6 |
| Code Editor | Monaco Editor (`@monaco-editor/react`) |
| Math Rendering | KaTeX (`react-katex`) |
| Icons | Lucide React |
| Styling | CSS Modules (no Tailwind) |
| Progress | localStorage via custom hook |

## Getting Started

```bash
npm install
npm run dev
```

Open http://localhost:5173

## Project Structure

```
src/
├── data/
│   └── curriculum.js        # All lesson content, exercises, blanks
├── components/
│   ├── Layout.jsx            # Sidebar + topbar + routing shell
│   ├── ContentSection.jsx    # Renders text/formula/callout/code/table
│   └── Exercise.jsx          # Fill-in-the-blank exercise with Monaco
├── pages/
│   ├── HomePage.jsx          # Landing page with chapter cards
│   └── LessonPage.jsx        # Individual lesson renderer
├── hooks/
│   └── useProgress.js        # localStorage progress tracking
├── utils/
│   └── answerValidator.js    # Flexible answer matching
├── App.jsx                   # Router + ProgressContext
└── index.css                 # Global CSS variables + theme
```

## Adding New Lessons

Edit `src/data/curriculum.js`. Each lesson has this structure:

```js
{
  id: "unique-id",
  title: "Lesson Title",
  slug: "url-slug",
  estimatedMinutes: 10,
  sections: [
    { type: "text", content: "Markdown-lite content..." },
    { type: "callout", variant: "info"|"warning"|"success", title: "...", content: "..." },
    { type: "formula", label: "...", latex: "...", explanation: "..." },
    { type: "code-snippet", label: "...", language: "python", code: "..." },
    { type: "comparison-table", headers: [...], rows: [[...], ...] },
  ],
  exercise: {
    id: "ex-unique-id",
    title: "Exercise Title",
    instructions: "What to do...",
    difficulty: "beginner"|"intermediate"|"advanced",
    starterCode: `python code with ___BLANK_1___ placeholders`,
    blanks: [
      {
        id: "BLANK_1",
        placeholder: "___BLANK_1___",
        hint: "Helpful hint...",
        acceptedAnswers: ["answer1", "answer2"],  // All accepted
        explanation: "Why this is correct..."
      }
    ],
    expectedOutput: "What the code should print..."
  }
}
```

## Chapters

1. 🐉 **Introduction to BDH** — What is BDH, Transformer limitations
2. ⚡ **Sparsity & k-WTA** — Metabolic efficiency, the pruning trajectory  
3. 🔬 **Monosemanticity** — Specialist neurons, Oja's Rule, Hebbian learning
4. 🧠 **Multi-Lobe Memory** — Spectral hierarchy, decay dynamics
5. 🧩 **Composable Intelligence** — Block diagonal merge, catastrophic forgetting
