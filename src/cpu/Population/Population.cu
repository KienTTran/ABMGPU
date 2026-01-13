//
// Created by kient on 6/17/2023.
//

#include "Population.cuh"
#include "../Model.h"
#include "Properties/PersonIndexGPU.h"
#include "Properties/PersonIndexByLocationStateAgeClass.h"
#include <glm/gtc/type_ptr.hpp>

Population::Population(Model* model) : model_(model){
};

Population::~Population(){
};


__global__ void update_person_position(int work_from, int work_to, int work_batch, float width, float height,
                                       glm::mat4 *buffer_person_models, glm::vec4 *buffer_person_colors,int rng_n,
                                       curandState *state){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (thread_index >= rng_n) return;  // critical
    curandState local_state = state[thread_index];
    for (int index = thread_index; index < work_batch; index += stride) {
        if(curand_uniform(&local_state) > 0.9f) {
            glm::mat4 model = buffer_person_models[index];
            glm::vec4 color = buffer_person_colors[index];
            float velocity = 0.0012;
            float x_n1_1 = (curand_uniform(&local_state) - 0.5f) * 2.0f;
            float y_n1_1 = (curand_uniform(&local_state) - 0.5f) * 2.0f;
            float x = x_n1_1 * width * velocity;
            float y = y_n1_1 * height * velocity;
            float rot = curand_uniform(&local_state) * 360.0f * velocity;
            model = translate(model, glm::vec3(x, y, 0.0f));
            model = translate(model, glm::vec3(0.0f, 0.0f, 1.0f));
            model = rotate(model, rot, glm::vec3(0.0f, 0.0f, 1.0f));
            model = translate(model, glm::vec3(0.0f, 0.0f, -1.0f));
            buffer_person_models[index] = model;
            buffer_person_colors[index] = color;
        }
        __syncthreads();
    }
    state[thread_index] = local_state;
}

__global__ void update_ogl_buffer(int work_from, int work_to, int work_batch,
                                  glm::mat4 buffer_person_models[],
                                  glm::vec4 buffer_person_colors[],
                                  glm::mat4 ogl_person_models[],
                                  glm::vec4 ogl_person_colors[]){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int index = thread_index; index < work_batch; index += stride) {
        ogl_person_models[work_from+index] = buffer_person_models[index];
        ogl_person_colors[work_from+index] = buffer_person_colors[index];
        __syncthreads();
    }
}

void Population::init(){
    person_index_list_ = new PersonIndexPtrList();
    auto p_index_to_render = new PersonIndexGPU();
    auto p_index_by_l_s_a = new PersonIndexByLocationStateAgeClass(Model::CONFIG->number_of_locations(),
                                                                   Person::NUMBER_OF_STATE,
                                                                   Model::CONFIG->number_of_age_classes());
    person_index_list_->push_back(p_index_to_render);
    person_index_list_->push_back(p_index_by_l_s_a);

    auto location_db = Model::CONFIG->location_db();

    number_of_locations = location_db.size();
    number_of_init_age_classes = Model::CONFIG->initial_age_structure().size();
    number_of_age_classes = Model::CONFIG->age_structure().size();
    h_cell_colors = thrust::host_vector<glm::vec4>(number_of_locations);

    //Random always init with max population size
    GPURandom::getInstance().init(Model::CONFIG->n_people_init() * Model::CONFIG->gpu_config().pre_allocated_mem_ratio,
                                  std::chrono::system_clock::now().time_since_epoch().count());

    printf("[Population] Population initial size: %d\n", Model::CONFIG->n_people_init());
    printf("[Population] Max processing people 1 batch: %d\n", Model::CONFIG->gpu_config().people_1_batch);

    setvbuf(stdout, NULL, _IONBF, 0);
}

void Population::initPop(){
    float width = Model::CONFIG->debug_config().width > 0 ? Model::CONFIG->debug_config().width : Model::CONFIG->render_config().window_width;
    float height = Model::CONFIG->debug_config().height > 0 ? Model::CONFIG->debug_config().height : Model::CONFIG->render_config().window_height;

    //Generate pop colors
    for(int loc_index = 0; loc_index < Model::CONFIG->location_db().size(); loc_index++){
        //Assign cell color by population in each cell
        if (thrust::get<3>(Model::CONFIG->location_db()[loc_index].asc_cell_data) > 1000000) {
            h_cell_colors[loc_index] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        }
        if (thrust::get<3>(Model::CONFIG->location_db()[loc_index].asc_cell_data) > 100000) {
            h_cell_colors[loc_index] = glm::vec4(1.0f, 0.5f, 0.0f, 1.0f);
        }
        else if (thrust::get<3>(Model::CONFIG->location_db()[loc_index].asc_cell_data) > 10000) {
            h_cell_colors[loc_index] = glm::vec4(0.5f, 1.0f, 0.0f, 1.0f);
        }
        else if (thrust::get<3>(Model::CONFIG->location_db()[loc_index].asc_cell_data) > 1000) {
            h_cell_colors[loc_index] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
        }
        else if (thrust::get<3>(Model::CONFIG->location_db()[loc_index].asc_cell_data) > 0){
            h_cell_colors[loc_index] = glm::vec4(0.0f, 0.5f, 0.5f, 1.0f);
        }
        else{
            h_cell_colors[loc_index] = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        }
//        h_cell_colors[loc_index] = glm::vec4(0.5f, 0.5f, 0.5f, 0.5f);
    }

    auto tp_start = std::chrono::high_resolution_clock::now();

    auto location_db = Model::CONFIG->location_db();

    int sum0 = 0;
    //Loop through all cells to init people
    for(int loc_index = 0; loc_index < Model::CONFIG->location_db().size(); loc_index++) {
        int temp_sum = 0;
        int last_temp_sum = 0;
        for (auto age_class = 0; age_class < number_of_init_age_classes; age_class++) {
            int number_of_individual_by_loc_age_class = static_cast<int>(Model::CONFIG->location_db()[loc_index].population_size *
                                                                        Model::CONFIG->location_db()[loc_index].age_distribution[age_class]);
//            printf("[Population] Init loc %d ac %d (%d + %d)",loc_index, age_class, number_of_individual_by_loc_age_class,temp_sum);
            if(Model::CONFIG->location_db()[loc_index].population_size > temp_sum + number_of_individual_by_loc_age_class){
                temp_sum += number_of_individual_by_loc_age_class;
                if(age_class == number_of_init_age_classes - 1) {
                    number_of_individual_by_loc_age_class = Model::CONFIG->location_db()[loc_index].population_size - temp_sum;
                }
            }
            else{
                if(age_class == number_of_init_age_classes - 1){
                    number_of_individual_by_loc_age_class = Model::CONFIG->location_db()[loc_index].population_size - temp_sum;
                    if(number_of_individual_by_loc_age_class < 0){
                        number_of_individual_by_loc_age_class = 0;
                    }
                }
                else{
                    int minus = (temp_sum + number_of_individual_by_loc_age_class) - Model::CONFIG->location_db()[loc_index].population_size;
                    number_of_individual_by_loc_age_class = number_of_individual_by_loc_age_class - minus;
                    temp_sum += number_of_individual_by_loc_age_class;
                }
            }
            last_temp_sum = temp_sum;
            for (auto i = 0; i < number_of_individual_by_loc_age_class; i++) {
                const auto age_from = (age_class == 0) ? 0 : Model::CONFIG->initial_age_structure()[age_class - 1];
                const auto age_to = Model::CONFIG->initial_age_structure()[age_class];
                Person *p = new Person();
                p->init();
                p->location = loc_index;
                p->set_host_state(Person::SUSCEPTIBLE);
                p->location_col = thrust::get<1>(Model::CONFIG->location_db()[loc_index].asc_cell_data);
                p->location_row = thrust::get<2>(Model::CONFIG->location_db()[loc_index].asc_cell_data);
                p->set_age(static_cast<const int&>(Model::RANDOM->random_uniform_int(age_from, age_to + 1)));
                //entity info
                //Set position follow .asc file
                float unit_x = width/(float)Model::CONFIG->asc_pop_ncols();
                float unit_y = height/(float)Model::CONFIG->asc_pop_nrows();
                float base_x_left = unit_x*p->location_col;
                float base_x_right = unit_x*p->location_col + unit_x;
                float base_y_bottom = unit_y*p->location_row;
                float base_y_top = unit_y*p->location_row + unit_y;
                float range_x = base_x_right - base_x_left;
                float range_y = base_y_top - base_y_bottom;
                float rand_x = Model::RANDOM->random_uniform_double(0.0,1.0);
                float rand_y = Model::RANDOM->random_uniform_double(0.0,1.0);
                float x = rand_x*range_x + base_x_left;
                float y = height - (rand_y*range_y + base_y_bottom);//OGL from bottom to ptop, so invert Y axis only
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, glm::vec3(x, y, 0.0f));
                p->model = model;
                p->color = h_cell_colors[loc_index];
                addPerson(p);
                sum0++;
            }
//            printf(" -> (%d %d) \n", number_of_individual_by_loc_age_class,temp_sum);
        }
//        printf("[Population] After init loc %d pop %d (%d)\n", loc_index,temp_sum, Model::CONFIG->location_db()[loc_index].population_size);
    }
//    printf("[Population] After init pop %d\n", sum0);

    auto lapse = std::chrono::high_resolution_clock::now() - tp_start;
    printf("[Population] Init population (%d %d) time: %d ms\n", sum0, Model::CONFIG->n_people_init(),
           std::chrono::duration_cast<std::chrono::milliseconds>(lapse).count());
    printf("\n");

    // Copy initial models and colors to persistent device vectors
    auto *pi = getPersonIndex<PersonIndexGPU>();
    d_person_models_.resize(pi->h_person_models().size());
    d_person_colors_.resize(pi->h_person_colors().size());
    thrust::copy(pi->h_person_models().begin(), pi->h_person_models().end(), d_person_models_.begin());
    thrust::copy(pi->h_person_colors().begin(), pi->h_person_colors().end(), d_person_colors_.begin());
}

void Population::performBirthEvent() {
    if (Model::CONFIG->debug_config().enable_debug_text) {
        std::cout << "[Population] Perform birth event"<< std::endl;
    }
    int birth_sum = 0;
    auto tp_start = std::chrono::high_resolution_clock::now();

    float width = Model::CONFIG->debug_config().width > 0 ? Model::CONFIG->debug_config().width : Model::CONFIG->render_config().window_width;
    float height = Model::CONFIG->debug_config().height > 0 ? Model::CONFIG->debug_config().height : Model::CONFIG->render_config().window_height;

    //Loop through all cells to add new people
    for(int loc_index = 0; loc_index < Model::CONFIG->location_db().size(); loc_index++) {
        auto poisson_means = Model::CONFIG->location_db()[loc_index].population_size * Model::CONFIG->birth_rate() / 365.0;
        const auto number_of_births = Model::RANDOM->random_poisson(poisson_means);
        if(number_of_births > 0){
            int person_id_in_age_class = 0;
            for(int p_index = 0; p_index < number_of_births; p_index++){
                Person *p = new Person();
                p->init();
                p->location = loc_index;
                p->set_host_state(Person::SUSCEPTIBLE);
                p->location_col = thrust::get<1>(Model::CONFIG->location_db()[loc_index].asc_cell_data);
                p->location_row = thrust::get<2>(Model::CONFIG->location_db()[loc_index].asc_cell_data);
                p->set_age(0);
                p->set_age_class(0);
                //entity info
                //Set position follow .asc file
                float unit_x = width/(float)Model::CONFIG->asc_pop_ncols();
                float unit_y = height/(float)Model::CONFIG->asc_pop_nrows();
                float base_x_left = unit_x*p->location_col;
                float base_x_right = unit_x*p->location_col + unit_x;
                float base_y_bottom = unit_y*p->location_row;
                float base_y_top = unit_y*p->location_row + unit_y;
                float range_x = base_x_right - base_x_left;
                float range_y = base_y_top - base_y_bottom;
                float rand_x = Model::RANDOM->random_uniform_double(0.0,1.0);
                float rand_y = Model::RANDOM->random_uniform_double(0.0,1.0);
                float x = rand_x*range_x + base_x_left;
                float y = height - (rand_y*range_y + base_y_bottom);//OGL from bottom to ptop, so invert Y axis only
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, glm::vec3(x, y, 0.0f));
                p->model = model;
                p->color = h_cell_colors[loc_index];
                addPerson(p);
                birth_sum++;
            }
            Model::CONFIG->location_db()[loc_index].population_size += number_of_births;
//            printf("[Population] Adding location %d %d pop_loc %d \n",loc_index, number_of_births, Model::CONFIG->location_db()[loc_index].population_size);
        }
    }

    auto lapse = std::chrono::high_resolution_clock::now() - tp_start;
    if(Model::CONFIG->debug_config().enable_debug_text){
        std::cout << "[Population] Update population birth (" << birth_sum<< ") event time: " << std::chrono::duration_cast<std::chrono::milliseconds>(lapse).count() << " ms"<< std::endl;
    }

    // Sync device vectors after births
    if (birth_sum > 0) {
        auto *pi = getPersonIndex<PersonIndexGPU>();
        d_person_models_.resize(pi->h_person_models().size());
        d_person_colors_.resize(pi->h_person_colors().size());
        thrust::copy(pi->h_person_models().begin(), pi->h_person_models().end(), d_person_models_.begin());
        thrust::copy(pi->h_person_colors().begin(), pi->h_person_colors().end(), d_person_colors_.begin());
    }
}

void Population::performDeathEvent() {
    if (Model::CONFIG->debug_config().enable_debug_text) {
        std::cout << "[Population] Perform death event"<< std::endl;
    }
    int dead_sum = 0;
    auto tp_start = std::chrono::high_resolution_clock::now();

    auto *pi = getPersonIndex<PersonIndexByLocationStateAgeClass>();
    auto& location_db = Model::CONFIG->location_db();
    for (auto loc_index = 0; loc_index < location_db.size(); loc_index++) {
        int loc_deaths = 0;
        for (auto hs = 0; hs < Person::NUMBER_OF_STATE - 1; hs++) {
            if (hs == Person::DEAD) continue;
            for (auto ac = 0; ac < Model::CONFIG->number_of_age_classes(); ac++) {
                const int size = pi->h_persons()[loc_index][hs][ac].size();
                if (size == 0) continue;
                auto poisson_means = size * Model::CONFIG->death_rate_by_age_class()[ac] / 365.0;
                assert(Model::CONFIG->death_rate_by_age_class().size() == Model::CONFIG->number_of_age_classes());
                const auto number_of_deaths = Model::RANDOM->random_poisson(poisson_means);
//                const auto number_of_deaths = floor(size*Model::CONFIG->death_rate_by_age_class()[ac]);
//                std::cout << fmt::format("[Population] Location {} state {} ac {} number of deaths: {}", loc, hs, ac, number_of_deaths);
                if (number_of_deaths == 0) continue;
                for (int i = 0; i < number_of_deaths; i++) {
                    // change state to Death;
                    const int index = Model::RANDOM->random_uniform(size);
                    auto *p_remove = pi->h_persons()[loc_index][hs][ac][index];
                    p_remove->set_host_state(Person::DEAD);
                }
                loc_deaths += number_of_deaths;
                dead_sum += number_of_deaths;
            }
        }
        location_db[loc_index].population_size -= loc_deaths;
//        printf("[Population] Removing location %d %d pop_loc %d \n",loc_index, loc_deaths, Model::CONFIG->location_db()[loc_index].population_size);
    }
    removeAllDeadPerson();

    auto lapse = std::chrono::high_resolution_clock::now() - tp_start;
    if(Model::CONFIG->debug_config().enable_debug_text){
        std::cout << "[Population] Update population death (" << dead_sum << ") event time: " << std::chrono::duration_cast<std::chrono::milliseconds>(lapse).count() << " ms"<< std::endl;
    }

    // Sync device vectors after deaths
    if (dead_sum > 0) {
        auto *pi = getPersonIndex<PersonIndexGPU>();
        d_person_models_.resize(pi->h_person_models().size());
        d_person_colors_.resize(pi->h_person_colors().size());
        thrust::copy(pi->h_person_models().begin(), pi->h_person_models().end(), d_person_models_.begin());
        thrust::copy(pi->h_person_colors().begin(), pi->h_person_colors().end(), d_person_colors_.begin());
    }
}

void Population::update(){
    if(Model::CONFIG->debug_config().enable_update){
        if(Model::CONFIG->debug_config().enable_debug_text){
            std::cout << "[Population] Perform movement event" << std::endl;
        }
        auto tp_start = std::chrono::high_resolution_clock::now();

        float width = Model::CONFIG->debug_config().width > 0 ? Model::CONFIG->debug_config().width : Model::CONFIG->render_config().window_width;
        float height = Model::CONFIG->debug_config().height > 0 ? Model::CONFIG->debug_config().height : Model::CONFIG->render_config().window_height;
        int n_threads = Model::CONFIG->gpu_config().n_threads;
        auto *pi = getPersonIndex<PersonIndexGPU>();
        int population_size = pi->h_persons().size();
        int batch_size = population_size;
        int batch_from = 0;
        int batch_to = batch_size;
        //This is to make sure threads fit all people in population
        n_threads = (batch_size < n_threads) ? batch_size : n_threads;

        // Run kernel directly on persistent device vectors
        update_person_position<<<((batch_size + n_threads - 1)/n_threads), n_threads>>>(batch_from, batch_to, batch_size, width, height,
                                                                                        thrust::raw_pointer_cast(d_person_models_.data()),
                                                                                        thrust::raw_pointer_cast(d_person_colors_.data()),
                                                                                        GPURandom::getInstance().rng_n,
                                                                                        GPURandom::getInstance().d_states);
        checkCudaErr(cudaDeviceSynchronize());
        checkCudaErr(cudaGetLastError());

        auto lapse = std::chrono::high_resolution_clock::now() - tp_start;
        if(Model::CONFIG->debug_config().enable_debug_text){
            std::cout << "[Population] Population size: " << pi->h_persons().size() << std::endl;
            std::cout << "[Population] Update population movement time: " << std::chrono::duration_cast<std::chrono::milliseconds>(lapse).count() << " ms" << std::endl;
        }
    }
}

void Population::updateRender(){
    auto tp_start = std::chrono::high_resolution_clock::now();

    auto *pi = getPersonIndex<PersonIndexGPU>();
    int n_threads = Model::CONFIG->gpu_config().n_threads;
    int population_size = pi->h_persons().size();
    int batch_size = population_size;
    int batch_from = 0;
    int batch_to = batch_size;
    //This is to make sure threads fit all people in population
    n_threads = (batch_size < n_threads) ? batch_size : n_threads;

    // Copy directly from persistent device vectors to OGL buffer
    update_ogl_buffer<<<((batch_size + n_threads - 1)/n_threads), n_threads>>>(batch_from, batch_to, batch_size,
                                                                               thrust::raw_pointer_cast(d_person_models_.data() + batch_from),
                                                                               thrust::raw_pointer_cast(d_person_colors_.data() + batch_from),
                                                                               Model::RENDER_ENTITY->d_ogl_buffer_model_ptr,
                                                                               Model::RENDER_ENTITY->d_ogl_buffer_color_ptr);
    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaGetLastError());

    auto lapse = std::chrono::high_resolution_clock::now() - tp_start;
    if(Model::CONFIG->debug_config().enable_debug_text) {
        std::cout << "[Population] Update population render (" << pi->h_persons().size() << " " << pi->h_person_models().size() << ") time: " << std::chrono::duration_cast<std::chrono::milliseconds>(lapse).count() << " ms" << std::endl;
    }
}

void Population::addPerson(Person* person) {
    for (PersonIndex* person_index : *person_index_list_) {
        person_index->add(person);
    }
    person->set_population(this);
}

void Population::notifyChange(Person* p, const Person::Property& property, const void* oldValue,
                              const void* newValue) {
    for (PersonIndex* person_index : *person_index_list_) {
        person_index->notifyChange(p, property, oldValue, newValue);
    }
}

void Population::removeAllDeadPerson() {
    // return all Death to object pool and clear vPersonIndex[l][dead][ac] for all location and ac
    auto *pi = getPersonIndex<PersonIndexByLocationStateAgeClass>();
    ThrustPersonPtrVectorHost removePersons;

    for (int loc = 0; loc < Model::CONFIG->number_of_locations(); loc++) {
        for (int ac = 0; ac < Model::CONFIG->number_of_age_classes(); ac++) {
            for (auto person : pi->h_persons()[loc][Person::DEAD][ac]) {
                removePersons.push_back(person);
            }
        }
    }

    for (Person* p : removePersons) {
        removeDeadPerson(p);
    }
}
void Population::removeDeadPerson(Person* person) {
    removePerson(person);
    ObjectHelpers::delete_pointer<Person>(person);
}
void Population::removePerson(Person* person) {
    for (PersonIndex* person_index : *person_index_list_) {
        person_index->remove(person);
    }
    person->set_population(nullptr);
}

void Population::start() {
    printf("[Population] Thread started\n");
    while (true) {
        update();
        if(Model::CONFIG->render_config().display_gui){
            updateRender();
        }
    }
}
