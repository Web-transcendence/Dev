import Fastify from 'fastify';
import { routes } from './routes.js';

// Load SSL certificates
// const httpsOptions = {
//     https: {
//         key: readFileSync('./key.pem'),      // Private key
//         cert: readFileSync('./cert.pem')     // Certificate
//     },
//     logger: true
// };
// const fastify = Fastify(httpsOptions);
// End of HTTPS setup

const fastify = Fastify({
    logger: true
})

fastify.register(routes)

fastify.listen({ host: '0.0.0.0', port: 3000 }, function (err, address) {
    if (err) {
        fastify.log.error(err)
        process.exit(1)
    }
    console.log(`Server is now listening on ${address}`)
})