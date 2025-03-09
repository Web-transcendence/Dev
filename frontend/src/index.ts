import { routes } from './routes.js';
import Fastify from 'fastify';
import fastifyStatic from '@fastify/static';
import {join} from "node:path";
import {env} from "./env.js";
import {readFileSync} from "node:fs";

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
    logger: {
        level: 'debug'
    }
})


// Reset database and id to 1
// DELETE FROM clients;
// VACUUM;
// DELETE FROM clients;
// DELETE FROM sqlite_sequence;


console.log('Base de données initialisée avec succès.');

// const clientSchema = z.object({
//     name: z.string().min(2, "Le nom doit contenir au moins 2 caractères"),
//     email: z.string().email("Email invalide"),
//     password: z.string().min(6, "Le mot de passe doit contenir au moins 6 caractères"),
// });

fastify.register(fastifyStatic, {
    root: join(import.meta.dirname, "..", "public"),
    prefix: "/static/"
})


fastify.get("/*", (req, res) => { // Route pour la page d'accueil
    console.log('test');
    const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
    const readFile = readFileSync(pagePath, 'utf8');
    res.type('text/html').send(readFile);
});


fastify.setNotFoundHandler((request, reply) => {
    const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
    const readFile = readFileSync(pagePath, 'utf8');
    reply.type('text/html').send(readFile);
});

fastify.register(routes)

fastify.listen({ host: '0.0.0.0', port: 3001 }, function (err, address) {
    if (err) {
        fastify.log.error(err)
        process.exit(1)
    }
    console.log(import.meta.dirname);
    console.log(`Server is now  listening on ${address}`)
})
