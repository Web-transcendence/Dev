import Fastify, { FastifyReply, FastifyRequest } from 'fastify'
import httpProxy from '@fastify/http-proxy'
import jwt from 'jsonwebtoken'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import fastifyStatic from '@fastify/static'
import { env } from './env'
import { routes } from './routes'

const httpsOptions = {
	https: {
		key: readFileSync(join(import.meta.dirname, '../../secure/key.pem')),      // Private key
		cert: readFileSync(join(import.meta.dirname, '../../secure/cert.pem'))     // Certificate
	},
}

const SECRET_KEY = process.env.SECRET_KEY

const app = Fastify(httpsOptions)

async function authentificate(req: FastifyRequest, reply: FastifyReply) {
	if (req.url === '/user-management/login' || req.url === '/user-management/register' || req.url === '/user-management/auth/google' || req.url === '/user-management/2faVerify')
		return
	try {
		const authHeader = req.headers.authorization
		if (!authHeader)
			return reply.status(401).send({ error: 'Unauthorized - No token provided' })

		const token = authHeader.split(' ')[1]
		if (!token)
			return reply.status(401).send({ error: 'Unauthorized - No token provided' })

		const decoded = jwt.verify(token, SECRET_KEY) as JwtPayload
		req.headers.id = decoded.id
	} catch (error) {
		return reply.status(401).send({ error: 'Unauthorized - invalid token' })
	}
}

app.get('/authJWT', async (req: FastifyRequest, res: FastifyReply) => {
	await authentificate(req, res)
	if (!req.headers.id)
		return res.status(401).send({ message: 'Unauthorized - No token provided' })
	return res.status(200).send({ message: 'Authentication successfull', id: req.headers.id })
})

app.register(fastifyStatic, {
	root: join(import.meta.dirname, '..', '..', 'public'),
	prefix: '/static/'
})

app.register(routes)

app.register(httpProxy, {
	upstream: 'http://pong:4443',
	prefix: '/pong',
	http2: false,
	preHandler: authentificate
})

app.register(httpProxy, {
	upstream: 'http://tower-defense:2246',
	prefix: '/tower-defense',
	http2: false,
	preHandler: authentificate
})

app.register(httpProxy, {
	upstream: 'http://user-management:5000',
	prefix: '/user-management',
	http2: false,
	preHandler: authentificate
})

app.register(httpProxy, {
	upstream: 'http://social:6500',
	prefix: '/social',
	http2: false,
	preHandler: authentificate
})

app.register(httpProxy, {
	upstream: 'http://tournament:7000',
	prefix: '/tournament',
	http2: false,
	preHandler: authentificate
})

app.register(httpProxy, {
	upstream: 'http://tower-defense:2246/ws',
	prefix: '/tower-defense/ws',
	websocket: true
})

app.register(httpProxy, {
	upstream: 'http://pong:4443/ws',
	prefix: '/pong/ws',
	websocket: true
})


app.get('/*', (req, res) => { // Route pour la page d'accueil
	const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, 'index.html')
	const readFile = readFileSync(pagePath, 'utf8')
	res.status(202).type('text/html').send(readFile)
})


app.listen({ port: 4000, host: '0.0.0.0' }, (err, adrr) => {
	if (err) {
		console.error(err)
		process.exit(1)
	}
	console.log(`server running on ${adrr}, ${join(import.meta.dirname, '../secure/cert.pem')}`)
})