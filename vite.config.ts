import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	ssr: {
		noExternal: ['three', '3d-force-graph']
	},
	server: {
		port: 5173,
		strictPort: true,
		proxy: {
			'/learn': 'http://localhost:5174',
			'/api': 'http://localhost:8000'
		}
	}
});
