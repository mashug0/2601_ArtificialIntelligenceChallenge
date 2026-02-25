/** @type {import('tailwindcss').Config} */
export default {
	darkMode: ['class'],
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			fontFamily: {
				sans: ['Space Grotesk', 'sans-serif'],
				serif: ['DM Sans', 'sans-serif'],
				mono: ['JetBrains Mono', 'monospace']
			},
			colors: {
				// Deep glow theme tokens
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				neon: {
					cyan: 'hsl(var(--neon-cyan))',
					magenta: 'hsl(var(--neon-magenta))',
					amber: 'hsl(var(--neon-amber))',
					emerald: 'hsl(var(--neon-emerald))'
				},
				surface: {
					DEFAULT: 'hsl(var(--surface))',
					hover: 'hsl(var(--surface-hover))'
				},
				sidebar: {
					DEFAULT: 'hsl(var(--sidebar-background))',
					foreground: 'hsl(var(--sidebar-foreground))',
					primary: 'hsl(var(--sidebar-primary))',
					'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
					accent: 'hsl(var(--sidebar-accent))',
					'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
					border: 'hsl(var(--sidebar-border))',
					ring: 'hsl(var(--sidebar-ring))'
				},
				// Backwards-compatible "paper" palette used by existing components
				paper: {
					bg: 'hsl(var(--background))',
					surface: 'hsl(var(--surface))',
					border: 'hsl(var(--border))',
					text: 'hsl(var(--foreground))',
					muted: 'hsl(var(--muted-foreground))'
				},

				// Visualization colors (kept from original theme)
				excitatory: {
					DEFAULT: '#d97706',
					light: '#fbbf24',
					dark: '#b45309'
				},
				inhibitory: {
					DEFAULT: '#2563eb',
					light: '#60a5fa',
					dark: '#1d4ed8'
				},
				specialist: {
					DEFAULT: '#10b981',
					light: '#34d399',
					dark: '#059669'
				},
				inactive: {
					DEFAULT: '#4b5563',
					light: '#6b7280',
					dark: '#374151'
				}
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: { height: '0' },
					to: { height: 'var(--radix-accordion-content-height)' }
				},
				'accordion-up': {
					from: { height: 'var(--radix-accordion-content-height)' },
					to: { height: '0' }
				},
				'pulse-glow': {
					'0%, 100%': { opacity: '0.4' },
					'50%': { opacity: '1' }
				},
				'scan-line': {
					'0%': { transform: 'translateY(0%)' },
					'100%': { transform: 'translateY(100%)' }
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
				'scan-line': 'scan-line 3s linear infinite',
				// original slow pulse
				'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite'
			}
		}
	},
	// `tailwindcss-animate` is optional and was causing a resolve error in this environment,
	// while our custom keyframes/animations are already defined above.
	// If you later want its utility classes, we can re-add it with a compatible setup.
	plugins: [require('@tailwindcss/typography')]
};
