const puppeteer = require('puppeteer');
const fs = require('fs');

async function cookiescrape(url) {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(url);
    //
    const [el] = await page.$x("//*[@id=\"cn-notice-text\"]");
    // console.log(el)
    const txt = await el.getProperty('textContent');
    // console.log(p)
    const text = await txt.jsonValue();

    const cookietxt = text;

    // console.log(cookietxt)

    //
    const [el2] = await page.$x("//*[@id=\"page\"]/section/div/p[2]");
    // console.log(el)
    const txtVeg = await el2.getProperty('textContent');
    // console.log(p)
    const textVeg = await txtVeg.jsonValue();

    const veggietxt = textVeg;


    fs.writeFile('./cookies/rockandice.com/.cookie', cookietxt, function (err) {
        if (err) {
            return console.log(err);
        }
        console.log(("Cookie Saved!"))
    });

    fs.writeFile('./veggies/rockandice.com/.veggie', veggietxt, function (err) {
        if (err) {
            return console.log(err);
        }
        console.log(("Veggie Saved!"))
    });


    // console.log(cookietxt)


    browser.close()

}


cookiescrape("https://rockandice.com/climbing-gear-tips/should-you-be-allowed-to-practice-lead-falls-in-the-gym/");