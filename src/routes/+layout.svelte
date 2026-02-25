<script lang="ts">
	import '../app.css';
	import { page } from '$app/stores';

	let mobileMenuOpen = false;

	const navItems = [
		{ href: '/', label: '3D Brain' },
		{ href: '/monosemanticity', label: 'Monosemanticity' },
		{ href: '/activation-atlas', label: 'Activation Atlas' },
		{ href: '/hebbian', label: 'Hebbian Synapses' },
		{ href: '/concepts', label: 'Concept Memory' },
	];

	const learnItems = [
		{ href: '/rope', label: 'ROPE' },
		{ href: '/relu-lowrank', label: 'ReLU Low-Rank' },
	];

	$: currentPath = $page.url.pathname;
</script>

<div class="min-h-screen flex flex-col bg-background text-foreground">
	<!-- Header -->
	<header
		class="sticky top-0 z-50 flex items-center justify-between px-4 py-2 border-b border-border/50 bg-background/90 backdrop-blur-xl"
	>
		<div class="flex items-center gap-1">
			<a href="/" class="flex items-center gap-2 mr-6">
				<img src="/logo.png" alt="BDH Logo" class="h-9 w-auto" />
				<span class="font-semibold text-sm tracking-wide">BDH</span>
			</a>

			<!-- Primary Nav -->
			<nav class="hidden lg:flex items-center gap-1">
				{#each navItems as item}
					<a
						href={item.href}
						class="px-3 py-1.5 rounded-md text-xs font-medium transition-all"
						class:bg-primary={currentPath === item.href}
						class:text-background={currentPath === item.href}
						class:text-glow-cyan={currentPath === item.href}
						class:text-muted-foreground={currentPath !== item.href}
						class:hover:text-foreground={currentPath !== item.href}
						class:hover:bg-secondary={currentPath !== item.href}
					>
						{item.label}
					</a>
				{/each}

				<span class="w-px h-5 bg-border/60 mx-1"></span>

				{#each learnItems as item}
					<a
						href={item.href}
						class="px-3 py-1.5 rounded-md text-xs font-medium transition-all"
						class:bg-accent={currentPath === item.href}
						class:text-accent-foreground={currentPath === item.href}
						class:text-glow-magenta={currentPath === item.href}
						class:text-muted-foreground={currentPath !== item.href}
						class:hover:text-foreground={currentPath !== item.href}
						class:hover:bg-secondary={currentPath !== item.href}
					>
						{item.label}
					</a>
				{/each}

				<span class="w-px h-5 bg-border/60 mx-1"></span>

				<a
					href="/learn"
					class="px-4 py-1.5 rounded-md text-xs font-semibold transition-all border"
					class:bg-primary={currentPath === '/learn'}
					class:text-background={currentPath === '/learn'}
					class:border-primary={currentPath === '/learn'}
					class:glow-cyan={currentPath === '/learn'}
					class:text-primary={currentPath !== '/learn'}
					class:border-border={currentPath !== '/learn'}
					class:hover:bg-primary={currentPath !== '/learn'}
					class:hover:text-background={currentPath !== '/learn'}
				>
					📖 Learn
				</a>

				<a
					href="/comparison"
					class="px-4 py-1.5 rounded-md text-xs font-semibold transition-all border"
					class:bg-accent={currentPath === '/comparison'}
					class:text-accent-foreground={currentPath === '/comparison'}
					class:border-accent={currentPath === '/comparison'}
					class:glow-magenta={currentPath === '/comparison'}
					class:text-accent={currentPath !== '/comparison'}
					class:border-border={currentPath !== '/comparison'}
					class:hover:bg-accent={currentPath !== '/comparison'}
				>
					⚔ VS Battle
				</a>

				<a
					href="/studio"
					class="px-4 py-1.5 rounded-md text-xs font-semibold transition-all border"
					class:bg-primary={currentPath === '/studio'}
					class:text-background={currentPath === '/studio'}
					class:border-primary={currentPath === '/studio'}
					class:glow-cyan={currentPath === '/studio'}
					class:text-primary={currentPath !== '/studio'}
					class:border-border={currentPath !== '/studio'}
					class:hover:bg-primary={currentPath !== '/studio'}
					class:hover:text-background={currentPath !== '/studio'}
				>
					🧪 Studio
				</a>
			</nav>
		</div>

		<!-- Mobile toggle -->
		<button
			class="lg:hidden p-2 rounded-md hover:bg-secondary text-muted-foreground hover:text-foreground"
			on:click={() => (mobileMenuOpen = !mobileMenuOpen)}
		>
			<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				{#if mobileMenuOpen}
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
				{:else}
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
				{/if}
			</svg>
		</button>
	</header>

	<!-- Mobile menu -->
	{#if mobileMenuOpen}
		<nav
			class="lg:hidden border-b border-border/60 bg-background/95 backdrop-blur-xl flex flex-col gap-1 px-4 pb-3 pt-2"
		>
			{#each [...navItems, ...learnItems] as item}
				<a
					href={item.href}
					class="px-3 py-2 text-sm rounded-md transition-colors"
					class:bg-primary={currentPath === item.href}
					class:text-background={currentPath === item.href}
					class:text-muted-foreground={currentPath !== item.href}
					class:hover:bg-secondary={currentPath !== item.href}
					on:click={() => (mobileMenuOpen = false)}
				>
					{item.label}
				</a>
			{/each}
			<a
				href="/learn"
				class="mt-1 px-3 py-2 text-sm rounded-md font-bold transition-colors border"
				class:bg-primary={currentPath === '/learn'}
				class:text-background={currentPath === '/learn'}
				class:border-primary={currentPath === '/learn'}
				class:text-primary={currentPath !== '/learn'}
				class:border-border={currentPath !== '/learn'}
				class:hover:bg-primary={currentPath !== '/learn'}
				on:click={() => (mobileMenuOpen = false)}
			>
				📖 Learn
			</a>
			<a
				href="/comparison"
				class="mt-1 px-3 py-2 text-sm rounded-md font-bold transition-colors border"
				class:bg-accent={currentPath === '/comparison'}
				class:text-accent-foreground={currentPath === '/comparison'}
				class:border-accent={currentPath === '/comparison'}
				class:text-accent={currentPath !== '/comparison'}
				class:border-border={currentPath !== '/comparison'}
				class:hover:bg-accent={currentPath !== '/comparison'}
				on:click={() => (mobileMenuOpen = false)}
			>
				⚔ VS Battle
			</a>
			<a
				href="/studio"
				class="mt-1 px-3 py-2 text-sm rounded-md font-bold transition-colors border"
				class:bg-primary={currentPath === '/studio'}
				class:text-background={currentPath === '/studio'}
				class:border-primary={currentPath === '/studio'}
				class:text-primary={currentPath !== '/studio'}
				class:border-border={currentPath !== '/studio'}
				class:hover:bg-primary={currentPath !== '/studio'}
				on:click={() => (mobileMenuOpen = false)}
			>
				🧪 Studio
			</a>
		</nav>
	{/if}

	<!-- Main content -->
	<main class="flex-1">
		<slot />
	</main>

</div>
