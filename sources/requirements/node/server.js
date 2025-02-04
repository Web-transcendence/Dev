// Import the framework and instantiate it
import Fastify from 'fastify'
const fastify = Fastify({
    logger: true
})

const { ADDRESS = '0.0.0.0', PORT = '8081' } = process.env;

fastify.get('/', async (request, reply) => {
    return { message: 'Hello world!' }
})

fastify.listen({ host: ADDRESS, port: parseInt(PORT, 10) }, (err, address) => {
    if (err) {
        console.error(err)
        process.exit(1)
    }
    console.log(`Server listening at ${address}`)
})