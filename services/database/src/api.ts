import Fastify, {FastifyReply, FastifyRequest} from "fastify";
import {createClient, emailExist} from "./clientBase.js";


const fastify = Fastify({
    logger: true
})

interface queryEmail {
    email: string
}

fastify.get('/email-existing', (req: FastifyRequest, res: FastifyReply)=> {
    const query = req.query as queryEmail;
    console.log("query");
    if (emailExist(query.email))
        return res.status(400).send();
    return res.status(200).send();
})

fastify.post('/addUser', createClient)


fastify.listen({ host: '0.0.0.0', port: 8003 }, function (err, address) {
    if (err) {
        fastify.log.error(err)
        process.exit(1)
    }
    console.log(`Server is now listening on ${address}`)
})