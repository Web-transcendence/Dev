import { getAvatar } from './user.js'
import { loadPart } from './insert.js'
import { sseConnection } from './serverSentEvent.js'
import { displayNotification } from './notificationHandler.js'
import { setupModalListeners } from './modal.js'

declare global {
	interface Window { // For Google authenticator
		CredentialResponse: (credit: { credential: string }) => Promise<void>;
	}
}

declare const tsParticles: any
declare const AOS: any

export let connected = false
export let path: string | null = null

window.addEventListener('popstate', async () => {
	if (!connected && window.location.pathname === '/profile' )
		location.replace('/home')
	else if ((window.location.pathname === '/connect' || window.location.pathname === '/login') && connected) {
		history.replaceState(null, '', '/home')
		await loadPart('/home')
	} else
		await loadPart(window.location.pathname)
})

const unhauthorizePath = [
	'/pongFriend',
	'/towerFriend',
	'/pongTournament',
	'/towerTournament',
]

document.addEventListener('DOMContentLoaded', async () => {
	constantButton() // Constant button on the Single Page Application
	setupModalListeners() // Setup global mobal
	AOS.init({ // animate slides on scroll
		once: true,
		duration: 800,
	})
	// Reconnect User
	const token = sessionStorage.getItem('token')
	if (token && await checkForToken()) {
		await getAvatar()
		handleConnection(true)
	} else
		handleConnection(false)
	// For Client Connection
	document.getElementById('avatar')?.addEventListener('click', () => {
		if (connected)
			document.getElementById('profile')?.click()
		else
			document.getElementById('connect')?.click()
	})
	path = sessionStorage.getItem('path')
	if (path && !(!connected && path === '/profile') && !unhauthorizePath.includes(path))
		await loadPart(path)
	else
		await loadPart('/home')
	await sseConnection()
})

tsParticles.load('tsparticles', {
	fullScreen: { enable: false },
	particles: {
		number: { value: 500 },
		size: { value: 2 },
		move: { enable: true, speed: 1 },
		opacity: { value: 0.5 },
		color: { value: '#ffffff' },
	},
	background: {
		color: '#000000',
	},
})

async function checkForToken(): Promise<boolean> {
	try {
		const token = sessionStorage.getItem('token')
		const response = await fetch(`/authJWT`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				'authorization': 'Bearer ' + token,
			},
		})
		if (!response.ok) {
			return false
		}
		return true
	} catch (error) {
		console.error(error)
		return false
	}
}

function constantButton() {
	//Duo Button
	document.getElementById('connect')?.addEventListener('click', (event: MouseEvent) => navigate('/connect', event))
	document.getElementById('profile')?.addEventListener('click', (event: MouseEvent) => navigate('/profile', event))
	//navigation page
	document.getElementById('home')?.addEventListener('click', async (event: MouseEvent) => navigate('/home', event))
	document.getElementById('pongMode')?.addEventListener('click', (event: MouseEvent) => navigate('/pongMode', event))
	document.getElementById('towerDefense')?.addEventListener('click', (event: MouseEvent) => navigate('/towerMode', event))
	document.getElementById('tournaments')?.addEventListener('click', async (event: MouseEvent) => {
		if (!connected) {
			displayNotification('You need to be connected to access this page')
			await navigate('/connect', event)
		}
		else await navigate('/tournaments', event)
	})
	document.getElementById('matchHistory')?.addEventListener('click', async (event: MouseEvent) => {
		if (!connected) {
			displayNotification('You need to be connected to access this page')
			await navigate('/connect', event)
		}
		else await navigate('/matchHistory', event)
	})
	// Footer
	document.getElementById('About')?.addEventListener('click', (event: MouseEvent) => navigate('/About', event))
}

export function handleConnection(input: boolean) {
	if (input) {
		document.getElementById('connect')?.classList.add('hidden')
		document.getElementById('profile')?.classList.remove('hidden')
	} else {
		sessionStorage.clear()
		document.getElementById('connect')?.classList.remove('hidden')
		document.getElementById('profile')?.classList.add('hidden')
	}
	connected = input
}

window.CredentialResponse = async (credit: { credential: string }) => {
	try {
		const response = await fetch(`/user-management/auth/google`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({ credential: credit.credential }),
		})
		if (!response.ok)
			console.error('Error: From Google UserManager')
		else {
			const reply = await response.json()
			if (reply.valid) {
				if (reply.avatar) sessionStorage.setItem('avatar', reply.avatar)
				if (reply.token) sessionStorage.setItem('token', reply.token)
				if (reply.id) sessionStorage.setItem('id', reply.id)
				if (reply.nickName) sessionStorage.setItem('nickName', reply.nickName)
				await navigate('/About')
				await getAvatar()
				if (reply.nickName.includes('googleNickname')) displayNotification('Your name is set by default, you can change it in the profile section')
				await sseConnection()
			}
		}
	} catch (error) {
		console.error('Error:', error)
	}
}

export async function navigate(newPath: string, event?: MouseEvent): Promise<void> {
	if (event) event.preventDefault()

	if (path === newPath) return
	path = newPath
	handleConnection(await checkForToken())
	if (!connected && newPath == '/profile')
		newPath = '/connect'
	if (connected && newPath == '/connect')
		newPath = '/profile'
	history.pushState({}, '', newPath)
	await loadPart(newPath)
}
