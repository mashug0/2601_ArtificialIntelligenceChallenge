import React, { useState } from 'react'
import { Copy, Check, ChevronDown, ChevronUp } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'
import styles from './ContentSection.module.css'

export default function ContentSection({ section }) {
  switch (section.type) {
    case 'text':       return <TextSection section={section} />
    case 'callout':    return <CalloutSection section={section} />
    case 'formula':    return <FormulaSection section={section} />
    case 'code-snippet': return <CodeSnippet section={section} />
    case 'comparison-table': return <ComparisonTable section={section} />
    default:           return null
  }
}

function TextSection({ section }) {
  // Parse **bold** and `inline code`
  const parsed = parseMarkdown(section.content)
  return (
    <div
      className={styles.textSection}
      dangerouslySetInnerHTML={{ __html: parsed }}
    />
  )
}

function parseMarkdown(text) {
  return text
    .split('\n\n')
    .map(para => {
      // Lists: lines starting with numbers or -
      const lines = para.trim().split('\n')
      const isList = lines.every(l => /^(\d+\.|-)/.test(l.trim()))
      if (isList) {
        const isOrdered = /^\d+\./.test(lines[0].trim())
        const tag = isOrdered ? 'ol' : 'ul'
        const items = lines
          .map(l => l.replace(/^(\d+\.|-)/, '').trim())
          .map(l => `<li>${formatInline(l)}</li>`)
          .join('')
        return `<${tag}>${items}</${tag}>`
      }
      return `<p>${formatInline(para.trim())}</p>`
    })
    .join('')
}

function formatInline(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
}

function CalloutSection({ section }) {
  const variants = {
    info:    { icon: 'ℹ️', color: 'var(--accent)', bg: 'var(--accent-glow)', border: 'rgba(0,212,255,0.25)' },
    warning: { icon: '⚠️', color: 'var(--amber)', bg: 'var(--amber-dim)',   border: 'rgba(255,179,71,0.25)' },
    success: { icon: '✅', color: 'var(--green)',  bg: 'var(--green-dim)',   border: 'rgba(74,222,128,0.25)' },
  }
  const v = variants[section.variant] || variants.info
  return (
    <div
      className={styles.callout}
      style={{ '--callout-color': v.color, '--callout-bg': v.bg, '--callout-border': v.border }}
    >
      <div className={styles.calloutTitle}>
        <span>{v.icon}</span>
        <strong>{section.title}</strong>
      </div>
      <p className={styles.calloutContent}>{section.content}</p>
    </div>
  )
}

function FormulaSection({ section }) {
  const [showExpl, setShowExpl] = useState(false)
  return (
    <div className={styles.formula}>
      <div className={styles.formulaLabel}>{section.label}</div>
      <div className={styles.formulaBox}>
        <BlockMath math={section.latex} />
      </div>
      {section.explanation && (
        <div>
          <button
            className={styles.formulaToggle}
            onClick={() => setShowExpl(v => !v)}
          >
            {showExpl ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            {showExpl ? 'Hide explanation' : 'Explain this formula'}
          </button>
          {showExpl && (
            <p className={styles.formulaExpl}>{section.explanation}</p>
          )}
        </div>
      )}
    </div>
  )
}

function CodeSnippet({ section }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(section.code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  // Simple Python syntax highlighter
  const highlighted = highlightPython(section.code)

  return (
    <div className={styles.codeSnippet}>
      <div className={styles.codeHeader}>
        <div className={styles.codeHeaderLeft}>
          <span className={styles.codeDot} style={{ background: '#ff5f57' }} />
          <span className={styles.codeDot} style={{ background: '#febc2e' }} />
          <span className={styles.codeDot} style={{ background: '#28c840' }} />
          <span className={styles.codeLabel}>{section.label}</span>
        </div>
        <button className={styles.copyBtn} onClick={handleCopy}>
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre
        className={styles.codeBody}
        dangerouslySetInnerHTML={{ __html: highlighted }}
      />
    </div>
  )
}

// Minimal Python syntax highlighter
function highlightPython(code) {
  const keywords = ['import', 'from', 'def', 'class', 'return', 'if', 'else', 'elif',
    'for', 'while', 'in', 'not', 'and', 'or', 'True', 'False', 'None',
    'with', 'as', 'try', 'except', 'raise', 'pass', 'lambda', 'self', 'super',
    'yield', 'assert', 'del', 'global', 'nonlocal', 'is', 'print']

  let result = escapeHtml(code)

  // Comments (do first to avoid re-processing)
  result = result.replace(/(#[^\n]*)/g, '<span class="hl-comment">$1</span>')
  // Strings
  result = result.replace(/("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g,
    '<span class="hl-string">$1</span>')
  // Numbers
  result = result.replace(/\b(\d+\.?\d*)\b/g, '<span class="hl-number">$1</span>')
  // Keywords
  keywords.forEach(kw => {
    result = result.replace(new RegExp(`\\b(${kw})\\b`, 'g'), '<span class="hl-keyword">$1</span>')
  })
  // Decorators
  result = result.replace(/(@\w+)/g, '<span class="hl-decorator">$1</span>')
  // Function defs
  result = result.replace(/\bdef\s+(\w+)/g, 'def <span class="hl-fn">$1</span>')
  result = result.replace(/\bclass\s+(\w+)/g, 'class <span class="hl-class">$1</span>')

  return result
}

function escapeHtml(text) {
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
}

function ComparisonTable({ section }) {
  return (
    <div className={styles.tableWrap}>
      <div className={styles.tableTitle}>{section.title}</div>
      <table className={styles.table}>
        <thead>
          <tr>
            {section.headers.map((h, i) => (
              <th key={i} className={styles.th}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {section.rows.map((row, ri) => (
            <tr key={ri} className={styles.tr}>
              {row.map((cell, ci) => (
                <td key={ci} className={`${styles.td} ${ci === 0 ? styles.tdLabel : ''}`}>
                  <span dangerouslySetInnerHTML={{ __html: formatInline(cell) }} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
