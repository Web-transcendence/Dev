import fastify from 'fastify'
import {User} from "./User.js";


const app = fastify();

app.post('/sign-up', async (req, res) => {
    try {
        const { name, email, password } = req.body as { name: string; email: string; password: string };

    const client: User = new User(name);

    const addRes = await client.addClient(email, password);
    if (!addRes.success) {
        return res.status(409).send({error: `${addRes.errorType} already exists`});
    }}
    catch(err) {
        return res.status(500).send({error: "Server error: ", err});
    }

    return res.status(201).send();
});

app.listen({port: 8001, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})