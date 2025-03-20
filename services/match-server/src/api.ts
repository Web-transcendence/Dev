import Fastify from 'fastify';
import fastifyWebsocket from '@fastify/websocket';
import fs from 'fs';
import path from 'path';
import {readFileSync} from "node:fs";
import {join} from "node:path";

const fastify = Fastify({
    https: {
        key: fs.readFileSync('./secure/key.pem'),
        cert: fs.readFileSync('./secure/cert.pem'),
    },
    logger: true
});

// const fastify = Fastify({ logger: true });

fastify.register(fastifyWebsocket);

fastify.register(import('@fastify/static'), {
    root: path.join(import.meta.dirname, '../public'),
    prefix: '/public/',
});


fastify.get('/ws', { websocket: true }, (socket, req) => {
    console.log('Client connecté');

    socket.on('message', message => {  // ✅ Utilisation correcte
        console.log('Message reçu:', message.toString());
        socket.send('Message reçu');
    });

    socket.on('close', () => {
        console.log('Client déconnecté');
    });
});

fastify.listen({ port: 8080, host: '0.0.0.0' }, (err, adrr) => {
    console.log("Chargement du certificat SSL...");
    console.log("Certificat trouvé ?", fs.existsSync('./secure/cert.pem'));
    console.log("Clé privée trouvée ?", fs.existsSync('./secure/key.pem'));
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
});