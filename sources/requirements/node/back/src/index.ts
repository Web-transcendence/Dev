import Fastify from 'fastify';
import {existsSync, readFileSync} from "node:fs";
import {join} from "node:path";

// import dbConnector from './our-db-connector.js';
import {env} from "./env.js";
import {z} from "zod";
import { routes } from './routes.js';

// Load SSL certificates
const httpsOptions = {
    https: {
        key: readFileSync('./key.pem'),      // Private key
        cert: readFileSync('./cert.pem')     // Certificate
    },
    logger: true
};

const fastify = Fastify(httpsOptions);
// const fastify = Fastify({
//     logger: true
// })

// fastify.register(dbConnector)
fastify.register(routes)

fastify.listen({ host: '0.0.0.0', port: 8443 }, function (err, address) {
    if (err) {
        fastify.log.error(err)
        process.exit(1)
    }
    console.log(`Server is now listening on ${address}`)
})