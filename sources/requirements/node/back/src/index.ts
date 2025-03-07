import { routes } from './routes.js';
import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import {join} from "node:path";
import {env} from "./env.js";
import {readFileSync} from "node:fs";
import {CreateClient} from './database.js';

// Load SSL certificates
const httpsOptions = {
    https: {
        key: readFileSync(join(import.meta.dirname, '../src/static/secure/key.pem')),      // Private key
        cert: readFileSync(join(import.meta.dirname, '../src/static/secure/cert.pem'))     // Certificate
    },
    logger: true
};
const fastify = Fastify(httpsOptions);

fastify.register(fastifyStatic, {
    root: join(import.meta.dirname, "..", "src", "static"),
    prefix: "/static/"
})

fastify.get("/*", (req, res) => { // Route pour la page d'accueil
    const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
    const readFile = readFileSync(pagePath, 'utf8');
    res.type('text/html').send(readFile);
});

fastify.register(routes)
fastify.register(CreateClient)

fastify.listen({ host: '127.0.0.1', port: 3001 }, function (err, address) {
    if (err) {
        fastify.log.error(err)
        process.exit(1)
    }
    console.log(`Server is now listening on ${address}`)
})
