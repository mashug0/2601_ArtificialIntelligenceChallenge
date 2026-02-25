import React, { useState, useEffect, useRef } from 'react'
import { Outlet, useLocation, Link, useNavigate } from 'react-router-dom'
import { curriculum, getAllLessons } from '../data/curriculum'
import { useProgressContext } from '../App'
import {
  ChevronRight, ChevronDown, CheckCircle2, Circle,
  Search, Menu, X, BookOpen, Zap
} from 'lucide-react'
import styles from './Layout.module.css'

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedChapters, setExpandedChapters] = useState({})
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const { isLessonComplete, getTotalCompleted, getTotalExercisesCompleted } = useProgressContext()
  const searchRef = useRef(null)

  const allLessons = getAllLessons()
  const totalLessons = allLessons.length

  // Auto-expand chapter of current lesson
  useEffect(() => {
    const parts = location.pathname.split('/').filter(Boolean)
    if (parts[0]) {
      setExpandedChapters(prev => ({ ...prev, [parts[0]]: true }))
    }
  }, [location.pathname])

  // Wire Ctrl+K / Cmd+K to focus search input
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        searchRef.current?.focus()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const toggleChapter = (chapterId) => {
    setExpandedChapters(prev => ({ ...prev, [chapterId]: !prev[chapterId] }))
  }

  const filteredCurriculum = searchQuery.trim()
    ? curriculum.map(chapter => ({
        ...chapter,
        lessons: chapter.lessons.filter(l =>
          l.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
          chapter.title.toLowerCase().includes(searchQuery.toLowerCase())
        )
      })).filter(c => c.lessons.length > 0)
    : curriculum

  const isActiveLesson = (chapterSlug, lessonSlug) =>
    location.pathname === `/${chapterSlug}/${lessonSlug}`

  const completedCount = getTotalCompleted()
  const progressPct = totalLessons > 0 ? Math.round((completedCount / totalLessons) * 100) : 0

  return (
    <div className={styles.root}>
      {/* ── TOP BAR ── */}
      <header className={styles.topbar}>
        <div className={styles.topbarLeft}>
          <button
            className={styles.menuBtn}
            onClick={() => setMobileOpen(o => !o)}
            aria-label="Toggle sidebar"
          >
            <Menu size={18} />
          </button>
          <Link to="/" className={styles.logo}>
            <img src="/logo.png" alt="BDH Logo" className={styles.logoImg} />
            <span className={styles.logoText}>BDH<span className={styles.logoDocs}>Docs</span></span>
          </Link>
        </div>

        <div className={styles.topbarRight}>
          <div className={styles.searchBox}>
            <Search size={14} className={styles.searchIcon} />
            <input
              ref={searchRef}
              className={styles.searchInput}
              placeholder="Search..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
            <kbd className={styles.searchKbd}>Ctrl K</kbd>
          </div>
        </div>
      </header>

      <div className={styles.body}>
        {/* ── SIDEBAR ── */}
        <aside className={`${styles.sidebar} ${mobileOpen ? styles.sidebarMobileOpen : ''}`}>
          <div className={styles.sidebarInner}>
            {/* Progress summary */}
            <div className={styles.progressCard}>
              <div className={styles.progressHeader}>
                <span className={styles.progressLabel}>
                  <Zap size={12} /> Progress
                </span>
                <span className={styles.progressPct}>{progressPct}%</span>
              </div>
              <div className={styles.progressBarTrack}>
                <div
                  className={styles.progressBarFill}
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              <div className={styles.progressStats}>
                <span>{completedCount}/{totalLessons} lessons</span>
                <span>{getTotalExercisesCompleted()} exercises</span>
              </div>
            </div>

            {/* Chapter/lesson tree */}
            <nav className={styles.nav}>
              {filteredCurriculum.map(chapter => {
                const isExpanded = expandedChapters[chapter.slug] !== false
                const chapterDone = chapter.lessons.every(l => isLessonComplete(l.id))

                return (
                  <div key={chapter.id} className={styles.chapterGroup}>
                    <button
                      className={styles.chapterBtn}
                      onClick={() => toggleChapter(chapter.slug)}
                    >
                      <span className={styles.chapterIcon}>{chapter.icon}</span>
                      <span className={styles.chapterTitle}>{chapter.title}</span>
                      {chapterDone && <CheckCircle2 size={12} className={styles.chapterDone} />}
                      {isExpanded
                        ? <ChevronDown size={14} className={styles.chevron} />
                        : <ChevronRight size={14} className={styles.chevron} />
                      }
                    </button>

                    {isExpanded && (
                      <div className={styles.lessonList}>
                        {chapter.lessons.map(lesson => {
                          const active = isActiveLesson(chapter.slug, lesson.slug)
                          const done = isLessonComplete(lesson.id)

                          return (
                            <Link
                              key={lesson.id}
                              to={`/${chapter.slug}/${lesson.slug}`}
                              className={`${styles.lessonLink} ${active ? styles.lessonLinkActive : ''} ${done ? styles.lessonLinkDone : ''}`}
                              onClick={() => setMobileOpen(false)}
                            >
                              <span className={styles.lessonDot}>
                                {done
                                  ? <CheckCircle2 size={13} />
                                  : <Circle size={13} />
                                }
                              </span>
                              <span className={styles.lessonTitle}>{lesson.title}</span>
                              {lesson.estimatedMinutes && (
                                <span className={styles.lessonMins}>{lesson.estimatedMinutes}m</span>
                              )}
                            </Link>
                          )
                        })}
                      </div>
                    )}
                  </div>
                )
              })}
            </nav>
          </div>
        </aside>

        {/* Mobile overlay */}
        {mobileOpen && (
          <div className={styles.overlay} onClick={() => setMobileOpen(false)} />
        )}

        {/* ── MAIN CONTENT ── */}
        <main className={styles.main}>
          <Outlet />
        </main>
      </div>
    </div>
  )
}
