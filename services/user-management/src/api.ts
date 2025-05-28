import fastify, { FastifyReply } from 'fastify'
import userRoutes from './routes.js'
import { googleAuth } from './googleApi.js'
import { FastifySSEPlugin } from 'fastify-sse-v2'

const app = fastify()

export const INTERNAL_PASSWORD = process.env.SECRET_KEY

if (!INTERNAL_PASSWORD) {
	throw new Error('INTERNAL_PASSWORD is not set in environment variables.')
}
app.register(FastifySSEPlugin)
app.register(userRoutes)
app.post('/auth/google', googleAuth)

export const connectedUsers = new Map<number, FastifyReply>()


app.listen({ port: 5000, host: '0.0.0.0' }, (err, adrr) => {
	if (err) {
		console.error(err)
		process.exit(1)
	}
	console.log(`server running on ${adrr}`)
})
