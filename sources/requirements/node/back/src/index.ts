import { routes } from './routes.js';
import Fastify from 'fastify';

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

// import Fastify from 'fastify';
// import path from 'path';
// import fastifyStatic from '@fastify/static';
// import {env} from "./env.js";
// import {readFileSync} from "node:fs";
// import {join} from "node:path";
//
// const fastify = Fastify({ logger: true });
//
// // Servir les fichiers statiques (CSS, JS, etc.)
// fastify.register(fastifyStatic, {
//     root: path.join(import.meta.dirname, env.TRANS_VIEWS_PATH),
//
//     prefix: '/views/', // URL prefix for static files
// });
//
// const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
//
// // Route principale qui sert l'index.html
// fastify.get('/*', async (_, res) => {
//     const file = readFileSync(pagePath, 'utf8');
//     res.raw.writeHead(200, {'Content-Type': 'text/html'});
//     res.raw.write(file);
// });
//
// fastify.register(routes)
// // Démarrage du serveur
// const start = async () => {
//     try {
//         await fastify.listen({ port: 3000, host: '0.0.0.0' });
//         console.log('Server is running on http://localhost:3000');
//     } catch (err) {
//         fastify.log.error(err);
//         process.exit(1);
//     }
// };
//
// start();
