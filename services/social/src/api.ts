import fastify from 'fastify'
import socialRoutes from './routes.js'

export const INTERNAL_PASSWORD = process.env.SECRET_KEY


const app = fastify()

app.register(socialRoutes)

app.listen({ port: 6500, host: '0.0.0.0' }, (err, adrr) => {
	if (err) {
		console.error(err)
		process.exit(1)
	}
	console.log(`server running on ${adrr}`)
})