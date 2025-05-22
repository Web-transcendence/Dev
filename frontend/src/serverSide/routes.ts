import {FastifyInstance, FastifyReply, FastifyRequest} from 'fastify'
import {readFileSync} from 'node:fs'
import {join} from 'node:path'
import {env} from './env.js'

export async function routes(fastify: FastifyInstance) {
	fastify.get('/front.js', (req, res) => {
		const frontPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, 'front.js')
		const file = readFileSync(frontPath, 'utf8')
		res.type('text/js').send(file)
		return
	})
	fastify.get('/pong.js', (req, res) => {
		const pongPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, 'pong.js')
		const file = readFileSync(pongPath, 'utf8')
		res.type('text/js').send(file)
		return
	})
	// ROAD OF TAG
	const htmlRoutes = [
		'about',
		'About',
		'logout',
		'login',
		'register',
		'connect',
		'contact',
		'profile',
		'home',
		'towerRemote',
		'towerMode',
		'towerWatch',
		'GameFriend',
		'pongRemote',
		'pongTournament',
		'pongLocal',
		'pongWatch',
    "pongVsia",
		'pongMode',
		'tournaments',
		'lobby',
		'factor',
		'matchHistory',
		'brackets',
	]

	htmlRoutes.forEach(route => {
		fastify.get(`/part/${route}`, (req, reply) => {
			let filename = route === 'logout' ? 'home.html' : `${route}.html`
			try {
				const fullPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, filename)
				const html = readFileSync(fullPath, 'utf8')
				reply.type('text/html').send(html)
			} catch (error) {
				reply.code(404).send(`Page ${filename} non trouvée`)
			}
		})
	})

	fastify.get('/part/:name', (req, reply) => {
		try {
			const fullPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, 'index.html')
			const html = readFileSync(fullPath, 'utf8')
			reply.type('text/html').send(html)
		} catch (error) {
			reply.code(500).send('Erreur serveur : index.html introuvable')
		}
	})

	// FAV ICON

	const allowedImages = new Set([
		'favicon.png',
		'favicon.ico',
		'fourRobots.png',
		'eightRobots.png',
		'sixteenRobots.png',
		'Biggest.png',
		'login.png',
		'logout.png',
		'BigLock.png',
		'solo.png',
		'vsia.png',
		'remote.png',
		'watchRobots.png',
	])
	fastify.get('/images/:imageName', async (req: FastifyRequest<{
		Params: { imageName: string }
	}>, reply: FastifyReply) => {
		const {imageName} = req.params

		if (!allowedImages.has(imageName)) {
			return reply.code(404).send('Image non autorisée')
		}

		try {
			const filePath = join(import.meta.dirname, env.TRANS_IMG_PATH, imageName)
			const fileData = readFileSync(filePath)
			reply.type('image/png').send(fileData)
		} catch {
			reply.code(404).send('Fichier non trouvé')
		}
	})

	//Pong assets
	fastify.get('/assets/pong/ballup.png', function (req, reply) {
		try {
			const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, 'pong/ballup.png')
			const tag = readFileSync(frontPath)
			reply.type('img/ico').send(tag)
		} catch (error) {
			reply.code(404).send('Fichier ballup non trouvé')
		}
	})
	fastify.get('/assets/pong/bardown.png', function (req, reply) {
		try {
			const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, 'pong/bardown.png')
			const tag = readFileSync(frontPath)
			reply.type('img/ico').send(tag)
		} catch (error) {
			reply.code(404).send('Fichier bardown non trouvé')
		}
	})
	fastify.get('/assets/pong/barup.png', function (req, reply) {
		try {
			const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, 'pong/barup.png')
			const tag = readFileSync(frontPath)
			reply.type('img/ico').send(tag)
		} catch (error) {
			reply.code(404).send('Fichier barup non trouvé')
		}
	})

	// TowerDefense
	fastify.get('/assets/tower-defense/:filename', async (req: FastifyRequest<{
		Params: { filename: string }
	}>, reply: FastifyReply) => {
		try {
			const {filename} = req.params
			const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, 'tower-defense', filename)
			const file = readFileSync(frontPath)
			reply.code(224).type('image/png').send(file)
		} catch (error) {
			reply.code(404).send(`Fichier ${req.params.filename} non trouvé`)
		}
	})
}