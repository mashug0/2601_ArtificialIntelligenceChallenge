import { scaleLinear, scaleSequential } from 'd3-scale';
import { interpolateRgb } from 'd3-interpolate';
import type { SignalType } from '$lib/types';

// Deep-glow color palette tuned to match dashboard
export const COLORS = {
	// Excitatory signals → neon amber/orange
	excitatory: '#f97316',
	excitatoryLight: '#fed7aa',
	excitatoryDark: '#c2410c',

	// Inhibitory signals → cyan/teal
	inhibitory: '#38bdf8',
	inhibitoryLight: '#a5f3fc',
	inhibitoryDark: '#0e7490',

	// Specialist neurons → bright emerald with strong glow
	specialist: '#22c55e',
	specialistGlow: 'rgba(34, 197, 94, 0.8)',

	// Inactive / background nodes
	inactive: '#4b5563',
	inactiveLight: '#9ca3af',

	// Scene & edge defaults (match deep-glow background)
	background: '#050510',
	edge: '#1f2937'
} as const;

// Activation to node size scale (2-20px radius)
export const sizeScale = scaleLinear().domain([0, 2]).range([2, 20]).clamp(true);

// Excitatory activation to color scale (gray -> amber)
export const excitatoryColorScale = scaleSequential()
	.domain([0, 2])
	.interpolator(interpolateRgb(COLORS.inactive, COLORS.excitatory));

// Inhibitory activation to color scale (gray -> blue)
export const inhibitoryColorScale = scaleSequential()
	.domain([0, 2])
	.interpolator(interpolateRgb(COLORS.inactive, COLORS.inhibitory));

// Edge weight to opacity scale
export const edgeOpacityScale = scaleLinear().domain([0, 1]).range([0.1, 0.8]).clamp(true);

/**
 * Get the color for a node based on its signal type and activation
 */
export function getNodeColor(
	signalType: SignalType,
	excitatoryActivation: number,
	inhibitoryActivation: number,
	isSpecialist: boolean
): string {
	if (isSpecialist && excitatoryActivation > 0.1) {
		return COLORS.specialist;
	}

	switch (signalType) {
		case 'excitatory':
			return excitatoryColorScale(excitatoryActivation) as string;
		case 'inhibitory':
			return inhibitoryColorScale(inhibitoryActivation) as string;
		case 'inactive':
		default:
			return COLORS.inactive;
	}
}

/**
 * Get the size for a node based on activation strength
 */
export function getNodeSize(excitatoryActivation: number, inhibitoryActivation: number): number {
	const maxActivation = Math.max(excitatoryActivation, inhibitoryActivation);
	return sizeScale(maxActivation);
}

/**
 * Determine signal type based on excitatory and inhibitory values
 */
export function getSignalType(
	excitatoryActivation: number,
	inhibitoryActivation: number,
	threshold: number = 0.05
): SignalType {
	if (excitatoryActivation > threshold && excitatoryActivation >= inhibitoryActivation) {
		return 'excitatory';
	}
	if (inhibitoryActivation > threshold) {
		return 'inhibitory';
	}
	return 'inactive';
}

/**
 * Get emissive intensity for specialist neurons
 */
export function getEmissiveIntensity(isSpecialist: boolean, activation: number): number {
	if (!isSpecialist) return 0;
	return Math.min(0.5, activation * 0.3);
}
