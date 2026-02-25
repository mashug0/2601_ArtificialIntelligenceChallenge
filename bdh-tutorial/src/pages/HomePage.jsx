import React from 'react'
import { Link } from 'react-router-dom'
import { curriculum, getAllLessons } from '../data/curriculum'
import { useProgressContext } from '../App'
import { ChevronRight, CheckCircle2, Circle, Zap, BookOpen, Award } from 'lucide-react'
import styles from './HomePage.module.css'

export default function HomePage() {
  const { isLessonComplete, getTotalCompleted, getTotalExercisesCompleted } = useProgressContext()
  const allLessons = getAllLessons()
  const totalLessons = allLessons.length
  const completed = getTotalCompleted()
  const exercisesDone = getTotalExercisesCompleted()

  return (
    <div className={styles.page}>
      {/* Hero */}
      <div className={styles.hero}>
        <div className={styles.heroInner}>
          <h1 className={styles.heroTitle}>
            Learn BDH
            <span className={styles.heroAccent}> Architecture</span>
          </h1>
          <p className={styles.heroDesc}>
            An interactive tutorial on the Baby Dragon Hatchling — a biologically plausible
            post-Transformer architecture featuring sparse activations, Hebbian learning,
            and composable intelligence. Build intuition through hands-on PyTorch exercises.
          </p>

          {/* Stats */}
          <div className={styles.statsRow}>
            <div className={styles.stat}>
              <div className={styles.statValue}>{completed}/{totalLessons}</div>
              <div className={styles.statLabel}>Lessons</div>
            </div>
            <div className={styles.statDivider} />
            <div className={styles.stat}>
              <div className={styles.statValue}>{exercisesDone}</div>
              <div className={styles.statLabel}>Exercises Done</div>
            </div>
            <div className={styles.statDivider} />
            <div className={styles.stat}>
              <div className={styles.statValue}>~{allLessons.reduce((s, l) => s + (l.estimatedMinutes || 0), 0)} min</div>
              <div className={styles.statLabel}>Total Time</div>
            </div>
          </div>

          {/* CTA */}
          <Link
            to={`/${allLessons[0].chapterSlug}/${allLessons[0].slug}`}
            className={styles.ctaBtn}
          >
            {completed > 0 ? 'Continue Learning' : 'Start Learning'}
            <ChevronRight size={17} />
          </Link>
        </div>

        {/* Abstract neuron visual */}
        <div className={styles.heroVisual} aria-hidden="true">
          <div className={styles.neuronGrid}>
            {Array.from({ length: 64 }).map((_, i) => (
              <div
                key={i}
                className={styles.neuron}
                style={{
                  opacity: Math.random() < 0.15 ? 1 : Math.random() * 0.15,
                  animationDelay: `${Math.random() * 3}s`,
                  animationDuration: `${2 + Math.random() * 3}s`,
                  background: Math.random() < 0.1 ? 'var(--accent)' : 'var(--text-muted)',
                }}
              />
            ))}
          </div>
          <div className={styles.neuronLabel}>k-WTA: 3% Active</div>
        </div>
      </div>

      {/* Chapter cards */}
      <div className={styles.chaptersSection}>
        <h2 className={styles.chaptersTitle}>Curriculum</h2>
        <div className={styles.chaptersGrid}>
          {curriculum.map(chapter => {
            const doneLessons = chapter.lessons.filter(l => isLessonComplete(l.id)).length
            const totalChapterLessons = chapter.lessons.length
            const chapterPct = Math.round((doneLessons / totalChapterLessons) * 100)
            const firstLesson = chapter.lessons[0]

            return (
              <Link
                key={chapter.id}
                to={`/${chapter.slug}/${firstLesson.slug}`}
                className={styles.chapterCard}
              >
                <div className={styles.chapterCardTop}>

                  <div className={styles.chapterProgress}>
                    {doneLessons === totalChapterLessons ? (
                      <CheckCircle2 size={15} color="var(--green)" />
                    ) : (
                      <span className={styles.chapterProgressText}>{doneLessons}/{totalChapterLessons}</span>
                    )}
                  </div>
                </div>
                <h3 className={styles.chapterCardTitle}>{chapter.title}</h3>
                <p className={styles.chapterCardDesc}>{chapter.description}</p>

                <div className={styles.chapterCardMeta}>
                  <div className={styles.chapterProgressBar}>
                    <div
                      className={styles.chapterProgressFill}
                      style={{ width: `${chapterPct}%` }}
                    />
                  </div>
                  <div className={styles.chapterLessons}>
                    {chapter.lessons.map(lesson => (
                      <div
                        key={lesson.id}
                        className={`${styles.lessonDot} ${isLessonComplete(lesson.id) ? styles.lessonDotDone : ''}`}
                        title={lesson.title}
                      />
                    ))}
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
      </div>

      {/* About section */}
      <div className={styles.about}>
        <div className={styles.aboutGrid}>
          <div className={styles.aboutCard}>
            <Zap size={22} color="var(--accent)" />
            <h3>Sparse Activations</h3>
            <p>Learn how k-WTA forces only 3% of neurons to fire at once, achieving 30× fewer FLOPs than dense Transformers.</p>
          </div>
          <div className={styles.aboutCard}>
            <BookOpen size={22} color="var(--amber)" />
            <h3>Hands-On Exercises</h3>
            <p>Fill-in-the-blank PyTorch code exercises based directly on the BDH paper's architecture and training procedures.</p>
          </div>
          <div className={styles.aboutCard}>
            <Award size={22} color="var(--green)" />
            <h3>Composable Intelligence</h3>
            <p>Understand how BDH specialists can be merged via Block Diagonal Assembly — solving catastrophic forgetting forever.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
