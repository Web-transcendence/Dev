import Fastify from 'fastify'

import dbConnector from './our-db-connector.js'
import firstRoute from './our-first-route.js'

/**
 * @type {import('fastify').FastifyInstance} Instance of Fastify
 */
const fastify = Fastify({
    logger: true
})
fastify.register(dbConnector)
fastify.register(firstRoute)

fastify.listen({ port: 8081 }, function (err, address) {
    if (err) {
        fastify.log.error(err)
        process.exit(1)
    }
    console.log(`Server is now listening on ${address}`);
})